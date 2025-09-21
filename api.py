from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------


class Facet(BaseModel):
    hkl: str
    gamma: float


class ShellLayer(BaseModel):
    material_cif: str
    aspect: List[float] = [1.0, 1.0, 1.0]
    facets: List[Facet] = []


class LigJob(BaseModel):
    smiles: str
    ratio: float = 1.0


class BuildOptions(BaseModel):
    radius_A: float
    core_cif_filename: str
    aspect: List[float] = [1.0, 1.0, 1.0]
    facets: List[Facet] = []
    shells: List[ShellLayer] = []

    # legacy (kept for compatibility elsewhere)
    ligand: Optional[str] = None
    surf_tol: Optional[float] = None

    # SMILES passivation
    cap_distribution: Optional[str] = "uniform"  # 'uniform' | 'segmented' | 'random'
    cap_anionic_jobs: Optional[List[LigJob]] = None  # replace Cl
    cap_cationic_jobs: Optional[List[LigJob]] = None  # replace Rb


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(
    title="QD_Builder API (Auto-Passivation)",
    version="5.3.6",
    description="Builds nanocrystals with automatic ligand selection for charge neutrality.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = Path(__file__).parent / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>index.html not found</h1><p>Make sure the index.html file is in the same directory as api.py.</p>",
            status_code=404,
        )
    return HTMLResponse(content=index_path.read_text())


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

LIG_PLACEHOLDERS = {"Cl", "Rb"}
LOG_KEEP = 300
_LOG_RE = re.compile(r"(Build(ing)?|Wulff|facet|shell|core|Atoms?|Total charge|Export|Passivation|Done|Warn|Error)$", re.I)
_SKIP_RE = re.compile(r"(downgrade|CN=|unique,\s*depth=|q=\d|neighbor)", re.I)


def _filter_log(text: str) -> str:
    out: List[str] = []
    last = None
    for raw in text.splitlines():
        line = raw.strip()
        if _SKIP_RE.search(line):
            continue
        if not _LOG_RE.search(line):
            continue
        if len(line) > 140:
            line = line[:140] + " …"
        if line == last:
            continue
        out.append(line)
        last = line
    return "\n".join(out[-LOG_KEEP:])


def _count_grouped_xyz(xyz_text: str, core_elems: set[str], shell_elems: set[str]) -> dict:
    counts = {"core": Counter(), "shell": Counter(), "ligand": Counter()}
    lines = xyz_text.splitlines()
    try:
        n = int(lines[0].strip())
    except Exception:
        n = 0

    for i in range(2, 2 + n):
        if i >= len(lines):
            break
        el = (lines[i].split() or ["?"])[0]
        if el in LIG_PLACEHOLDERS:
            grp = "ligand"
        elif el in core_elems:
            grp = "core"
        elif el in shell_elems:
            grp = "shell"
        else:
            grp = "shell"
        counts[grp][el] += 1

    total = sum(sum(c.values()) for c in counts.values())
    fmt = lambda c: {"total": sum(c.values()), "by_element": dict(c)}  # noqa: E731
    return {
        "total_atoms": total,
        "core": fmt(counts["core"]),
        "shell": fmt(counts["shell"]),
        "ligand": fmt(counts["ligand"]),
    }


def _charge_of_xyz(xyz_text: str, charges: Dict[str, float]) -> float:
    lines = xyz_text.splitlines()
    try:
        n = int(lines[0].strip())
    except Exception:
        return 0.0

    total = 0.0
    for i in range(2, 2 + n):
        if i >= len(lines):
            break
        el = (lines[i].split() or ["?"])[0]
        total += charges.get(el, 0.0)
    return total


def run_cmd(cmd: List[str], cwd: Path) -> Tuple[str, str]:
    if cmd[0] == "nc-builder":
        exe = shutil.which("nc-builder")
        if not exe:
            raise RuntimeError("nc-builder not found on PATH for this server process.")
        cmd[0] = exe
    p = subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True)
    return p.stdout, p.stderr


def parse_cif_oxidation_numbers(text: str) -> Dict[str, float]:
    lines = text.splitlines()
    i = 0
    charges: Dict[str, float] = {}

    def read_loop(start: int):
        hdrs, rows, j = [], [], start + 1
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            hdrs.append(lines[j].strip())
            j += 1
        while j < len(lines):
            t = lines[j].strip()
            if not t or t.startswith(("loop_", "data_", "_")):
                break
            rows.append(t)
            j += 1
        return hdrs, rows, j

    _KEY_ALIASES = {
        "_atom_type.symbol": "_atom_type_symbol",
        "_atom_type.oxidation_number": "_atom_type_oxidation_number",
        "_atom_type.charge": "_atom_type_charge",
        "_atom_site.type_symbol": "_atom_site_type_symbol",
        "_atom_site.oxidation_number": "_atom_site_oxidation_number",
        "_atom_site.charge": "_atom_site_charge",
        "_atom_site.label": "_atom_site_label",
    }

    def _norm_key(k: str) -> str:
        return _KEY_ALIASES.get(k.strip().lower(), k.strip().lower())

    def _elem_from_label(s: str) -> str:
        m = re.match(r"([A-Z][a-z]?)", s)
        return m.group(1) if m else s

    def _parse_charge_str(v: str) -> float:
        v = str(v).strip().replace(",", ".").replace(" ", "")
        if v.endswith("+"):
            v = v[:-1]
        elif v.endswith("-"):
            v = "-" + v[:-1]
        if v.startswith("+"):
            v = v[1:]
        return float(v) if v else 0.0

    while i < len(lines):
        if lines[i].strip().lower().startswith("loop_"):
            hdrs, rows, j = read_loop(i)
            headers = [_norm_key(h) for h in hdrs]
            sym_candidates = ["_atom_type_symbol", "_atom_site_type_symbol", "_atom_site_label"]
            ox_candidates = [
                "_atom_type_oxidation_number",
                "_atom_site_oxidation_number",
                "_atom_type_charge",
                "_atom_site_charge",
            ]
            si = next((headers.index(k) for k in sym_candidates if k in headers), None)
            oi = next((headers.index(k) for k in ox_candidates if k in headers), None)
            if si is not None and oi is not None:
                for row in rows:
                    toks = re.findall(r"(?:'[^']*'|\"[^\"]*\"|\S+)", row)
                    if len(toks) <= max(si, oi):
                        continue
                    raw_sym, raw_ox = toks[si].strip("'\""), toks[oi].strip("'\"")
                    if raw_sym in ".?" or raw_ox in ".?":
                        continue
                    sym = _elem_from_label(raw_sym)
                    try:
                        charges[sym] = _parse_charge_str(raw_ox)
                    except (ValueError, IndexError):
                        continue
            i = j
        else:
            i += 1

    return charges


def execute_builder_command(
    cmd: List[str], out_dir: str, primary_output_filename: str
) -> Tuple[bool, str, Dict[str, Path | None]]:
    out_dir_path = Path(out_dir)
    final_path = out_dir_path / primary_output_filename
    output_stem = final_path.stem
    cut_path = out_dir_path / f"{output_stem}_cut.xyz"

    logging.info(f"Running command in {out_dir_path}: {' '.join(cmd)}")

    try:
        p = subprocess.run(
            cmd,
            cwd=str(out_dir_path),
            capture_output=True,
            env=os.environ.copy(),
            check=False,
            text=True,
        )
        stdout = p.stdout or ""
        stderr = p.stderr or ""
        log_output = stdout + ("\n" + stderr if stderr else "")

        ok = p.returncode == 0 and final_path.exists() and final_path.stat().st_size > 0
        if not ok:
            logging.error(
                f"nc-builder execution check failed (return code: {p.returncode}).\nLogs:\n{log_output}"
            )

        found_files = {
            "final": final_path if final_path.exists() else None,
            "cut": cut_path if cut_path.exists() else None,
        }
        return ok, log_output, found_files

    except FileNotFoundError:
        msg = "Error: 'nc-builder' command not found. Is your conda environment activated?"
        logging.error(msg)
        return False, msg, {}

    except Exception as e:
        msg = f"An unexpected error occurred: {e}"
        logging.error(msg, exc_info=True)
        return False, msg, {}


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def format_facets_to_dict(facets: List[Facet]) -> Dict[str, float]:
    return {f.hkl: f.gamma for f in facets}


def _dist_string(jobs: List[LigJob], mode: str) -> str:
    xs = [f"{float(max(0.0, j.ratio)):.6g}" for j in jobs if j.smiles.strip()]
    if not xs:
        xs = ["1"]
    mode = (mode or "uniform").lower()
    if mode not in {"uniform", "segmented", "random"}:
        mode = "uniform"
    return ":".join(xs + [mode])

def run_capper_cli(script_path: Path, xyz_text: str, jobs: List[LigJob],
                   dummy: str, dist_mode: str, workdir: Path) -> Tuple[str, str, str]:
    """Return (capped_xyz, cap_log, out_name)."""
    qd_path = workdir / "qd_for_capping.xyz"
    qd_path.write_text(xyz_text, encoding="utf-8")

    out_prefix = workdir / f"capped_{dummy.lower()}"
    smiles = [j.smiles.strip() for j in jobs if j.smiles.strip()]
    if not smiles:
        return xyz_text, "[cap] No ligands provided; skipping.", ""

    cmd = [sys.executable, str(script_path), "--qd", str(qd_path),
           "--out_prefix", str(out_prefix), "--job-ligands", *smiles,
           "--job-dummy", dummy, "--job-dist", _dist_string(jobs, dist_mode)]
    p = subprocess.run(cmd, cwd=str(workdir), text=True, capture_output=True)
    log = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")

    candidates = [f"{out_prefix}_final.xyz", f"{out_prefix}.xyz", f"{out_prefix}_capped.xyz"]
    xyz_out, out_name = None, ""
    for cand in candidates:
        cp = Path(cand)
        if cp.exists() and cp.stat().st_size > 0:
            xyz_out = cp.read_text(encoding="utf-8")
            out_name = cp.name
            break

    if p.returncode != 0 or not xyz_out:
        raise RuntimeError(f"Capper failed (rc={p.returncode}).\n{log}")
    return xyz_out, log, out_name
# ---------------------------------------------------------------------
# Route: build
# ---------------------------------------------------------------------
@app.post("/api/build", response_class=JSONResponse)
async def build_nanocrystal(files: List[UploadFile] = File(...), options: str = Form(...)):
    try:
        opts = BuildOptions.parse_raw(options)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {e}")

    tmpdir = tempfile.mkdtemp(prefix="qdb_")
    logging.info(f"Temporary directory created at: {tmpdir}")

    try:
        tmp_path = Path(tmpdir)

        # Save uploads
        file_map: Dict[str, Path] = {}
        for uploaded_file in files:
            s_filename = safe_filename(uploaded_file.filename)
            file_path = tmp_path / s_filename
            file_path.write_bytes(await uploaded_file.read())
            file_map[s_filename] = file_path

        # Core
        core_path = file_map.get(safe_filename(opts.core_cif_filename))
        if not core_path:
            raise HTTPException(404, f"Core file '{opts.core_cif_filename}' not found.")

        charges = parse_cif_oxidation_numbers(core_path.read_text("utf-8", "ignore"))
        core_elements = set(charges.keys())
        shell_elements: set[str] = set()

        import yaml

        # ---------------- First pass (info) ----------------
        logging.info("Running first pass to gather info...")
        first_pass_charges = charges.copy()
        if "Cl" not in first_pass_charges:
            first_pass_charges["Cl"] = -1.0

        temp_yaml_dict = {
            "shape": {"aspect": opts.aspect},
            "facets": [f.dict() for f in opts.facets],
            "charges": first_pass_charges,
            "passivation": {"ligand": "Cl", "surf_tol": 2.0},
        }
        temp_yaml_path = tmp_path / "temp_config.yml"
        temp_yaml_path.write_text(yaml.safe_dump(temp_yaml_dict, sort_keys=False), encoding="utf-8")

        first_pass_output_file = "final.xyz"
        first_pass_cmd = [
            "nc-builder",
            str(core_path.resolve()),
            str(temp_yaml_path.resolve()),
            "-r",
            f"{opts.radius_A:.4f}",
            "-o",
            str((tmp_path / first_pass_output_file).resolve()),
            "--center",
            "--write-all",
            "--verbose",
        ]
        p1_ok, p1_log, _ = execute_builder_command(first_pass_cmd, tmpdir, first_pass_output_file)
        if not p1_ok:
            return JSONResponse(status_code=500, content={"status": "failed", "log": p1_log})

        # ---------------- Second pass (final build) ----------------
        final_charges = charges.copy()
        if "Cl" not in final_charges:
            final_charges["Cl"] = -1.0
        if "Rb" not in final_charges:
            final_charges["Rb"] = 1.0

        passivation_block = {"ligand": "Cl", "surf_tol": 2.0}

        if not opts.shells:
            yaml_dict = {
                "shape": {"aspect": opts.aspect},
                "facets": [f.dict() for f in opts.facets],
                "charges": final_charges,
                "passivation": passivation_block,
            }
            cmd = ["nc-builder", str(core_path.resolve())]
        else:
            materials = [
                {
                    "name": "core",
                    "cif": str(core_path.resolve()),
                    "facets": format_facets_to_dict(opts.facets),
                    "shape": {"aspect": opts.aspect},
                }
            ]
            outermost_shell_path = core_path

            for i, shell in enumerate(opts.shells):
                shell_cif_path = file_map.get(safe_filename(shell.material_cif))
                if not shell_cif_path:
                    raise HTTPException(404, f"Shell material '{shell.material_cif}' not found.")

                materials.append(
                    {
                        "name": f"shell{i+1}",
                        "cif": str(shell_cif_path.resolve()),
                        "facets": format_facets_to_dict(shell.facets) or "inherit",
                        "shape": {"aspect": shell.aspect},
                    }
                )

                shell_charges = parse_cif_oxidation_numbers(shell_cif_path.read_text("utf-8", "ignore"))
                final_charges.update(shell_charges)
                shell_elements.update(shell_charges.keys())
                outermost_shell_path = shell_cif_path

            yaml_dict = {
                "materials": materials,
                "charges": final_charges,
                "symmetry": {"proper_rotations_only": True},
                "facet_options": {"pair_opposites": True},
                "passivation": passivation_block,
            }
            cmd = ["nc-builder", str(outermost_shell_path.resolve())]

        final_yaml_path = tmp_path / "config.yml"
        final_yaml_path.write_text(yaml.safe_dump(yaml_dict, sort_keys=False), encoding="utf-8")
        logging.info(f"Generated FINAL YAML config:\n{final_yaml_path.read_text()}")

        final_output_file = "final.xyz"
        cmd.extend(
            [
                str(final_yaml_path.resolve()),
                "-r",
                f"{opts.radius_A:.4f}",
                "-o",
                str((tmp_path / final_output_file).resolve()),
                "--write-all",
                "--center",
                "--verbose",
            ]
        )
        if opts.shells:
            cmd.extend(["--core-lattice-fit", "--core-strain-width", "2.5", "--core-center", "com"])

        ok, log, out_files = execute_builder_command(cmd, tmpdir, final_output_file)

        xyz_passivated_content = out_files.get("final").read_text() if out_files.get("final") else None
        xyz_unpassivated_content = out_files.get("cut").read_text() if out_files.get("cut") else None

        full_log = p1_log + "\n--- Second Pass ---\n" + log
        raw_log = full_log

        if not ok:
            return JSONResponse(status_code=500, content={"status": "failed", "log": raw_log})

        elements, total_charge = "unknown", 0
        grouped_counts: Dict[str, Any] = {}

        if xyz_passivated_content:
            try:
                from ase.io import read as ase_read

                atoms = ase_read(io.StringIO(xyz_passivated_content), format="xyz")
                symbols = atoms.get_chemical_symbols()
                elements = ",".join(sorted(set(symbols)))
                total_charge = sum(final_charges.get(s, 0.0) for s in symbols)
                grouped_counts = _count_grouped_xyz(xyz_passivated_content, core_elements, shell_elements)
            except Exception as e:
                logging.error(f"Could not parse passivated XYZ or calculate charge: {e}")

        # ---------- Organic passivation from SMILES (optional) ----------
        CAP_SCRIPT = Path(__file__).parent / "attach" / "attach_from_smiles.py"
        cap_logs: List[str] = []
        current_xyz = xyz_passivated_content or xyz_unpassivated_content or ""
        download_name = None  # <— add this
        try:
            if opts.cap_anionic_jobs:
                current_xyz, log1, name1 = run_capper_cli(
                    CAP_SCRIPT, current_xyz, opts.cap_anionic_jobs or [],
                    dummy="Cl", dist_mode=(opts.cap_distribution or "uniform"),
                    workdir=tmp_path
                )
                cap_logs.append(log1)
                download_name = name1 or download_name
        
            if opts.cap_cationic_jobs:
                current_xyz, log2, name2 = run_capper_cli(
                    CAP_SCRIPT, current_xyz, opts.cap_cationic_jobs or [],
                    dummy="Rb", dist_mode=(opts.cap_distribution or "uniform"),
                    workdir=tmp_path
                )
                cap_logs.append(log2)
                download_name = name2 or download_name
        
            if cap_logs:
                xyz_passivated_content = current_xyz
                # recompute metadata on capped structure
                from ase.io import read as ase_read

                atoms = ase_read(io.StringIO(current_xyz), format="xyz")
                symbols = atoms.get_chemical_symbols()
                elements = ",".join(sorted(set(symbols)))
                total_charge = sum(final_charges.get(s, 0.0) for s in symbols)
                grouped_counts = _count_grouped_xyz(current_xyz, core_elements, shell_elements)
                raw_log += "\n--- Organic capping ---\n" + "\n".join(cap_logs)

        except Exception as e:
            logging.exception("Capping step failed")
            raw_log += f"\n[cap][error] {e}"
        # -----------------------------------------------------------------

        # ---- compact summary ----
        init_charge = 0.0
        if xyz_unpassivated_content:
            init_charge = _charge_of_xyz(xyz_unpassivated_content, final_charges)

        cl_n = grouped_counts.get("ligand", {}).get("by_element", {}).get("Cl", 0)
        rb_n = grouped_counts.get("ligand", {}).get("by_element", {}).get("Rb", 0)
        core_comp = grouped_counts.get("core", {}).get("by_element", {})
        shell_comp = grouped_counts.get("shell", {}).get("by_element", {})

        summary_lines = [
            "Summary:",
            f"- Atoms total: {grouped_counts.get('total_atoms', 'NA')}",
            f"- Core:  {grouped_counts.get('core', {}).get('total', 0)}  | "
            + " ".join(f"{k}:{v}" for k, v in sorted(core_comp.items())),
            f"- Shell: {grouped_counts.get('shell', {}).get('total', 0)} | "
            + " ".join(f"{k}:{v}" for k, v in sorted(shell_comp.items())),
            f"- Ligand placeholders: Cl={cl_n}, Rb={rb_n}",
            f"- Charge: initial≈{round(init_charge)}, final≈{round(total_charge)}",
        ]
        raw_log = (raw_log + "\n" + "\n".join(summary_lines)).strip()

        return JSONResponse(
            content={
                "status": "success",
                "log": raw_log,
                "elements": elements,
                "xyz_passivated": xyz_passivated_content,
                "xyz_unpassivated": xyz_unpassivated_content,
                "total_charge": round(total_charge),
                "grouped_counts": grouped_counts,
                "download_name": download_name or "final.xyz"
            }
        )

    finally:
        logging.info(f"Cleaning up temporary directory: {tmpdir}")
        # shutil.rmtree(tmpdir)

class CapOnlyOptions(BaseModel):
    cap_distribution: Optional[str] = "uniform"
    cap_anionic_jobs: Optional[List[LigJob]] = None
    cap_cationic_jobs: Optional[List[LigJob]] = None

from fastapi import UploadFile, File, Form

@app.post("/api/passivate", response_class=JSONResponse)
async def passivate_nanocrystal(
    xyz_file: UploadFile = File(...),
    options: str = Form(...),
):
    opts = CapOnlyOptions.parse_raw(options)
    tmpdir = tempfile.mkdtemp(prefix="qdb_cap_")
    tmp_path = Path(tmpdir)
    try:
        # read uploaded XYZ
        raw = await xyz_file.read()
        base_xyz = raw.decode("utf-8", "ignore")

        CAP_SCRIPT = Path(__file__).parent / "attach" / "attach_from_smiles.py"
        current_xyz = base_xyz
        cap_logs, download_name = [], None

        if opts.cap_anionic_jobs:
            current_xyz, log1, name1 = run_capper_cli(
                CAP_SCRIPT, current_xyz, opts.cap_anionic_jobs or [],
                dummy="Cl", dist_mode=(opts.cap_distribution or "uniform"),
                workdir=tmp_path
            )
            cap_logs.append(log1); download_name = name1 or download_name

        if opts.cap_cationic_jobs:
            current_xyz, log2, name2 = run_capper_cli(
                CAP_SCRIPT, current_xyz, opts.cap_cationic_jobs or [],
                dummy="Rb", dist_mode=(opts.cap_distribution or "uniform"),
                workdir=tmp_path
            )
            cap_logs.append(log2); download_name = name2 or download_name

        return JSONResponse(content={
            "status": "success",
            "log": "\n".join(cap_logs),
            "xyz_passivated": current_xyz,
            "download_name": download_name or "capped_final.xyz",
        })
    finally:
        logging.info(f"Cleaning up temporary directory: {tmpdir}")
        # shutil.rmtree(tmpdir)
# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

