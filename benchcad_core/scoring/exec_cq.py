"""Run model-generated CadQuery code → write a STEP file.

Two helpers:
    extract_code(raw)         pull python from a fenced block
    execute_cq_to_step(code, step_path, timeout=300)
        runs the code in a subprocess; if no .exportStep is present we append one
        for `result`. Raises RuntimeError on subprocess failure or missing output.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_FENCE = re.compile(r"```(?:python|py|cadquery)?\s*\n(.*?)```", re.DOTALL)


def extract_code(raw: str) -> str:
    """First ```python ... ``` block, or whole text if no fence."""
    m = _FENCE.search(raw or "")
    if m:
        return m.group(1).strip()
    if raw and ("import cadquery" in raw or "cq.Workplane" in raw):
        return raw.strip()
    return ""


_OCP_HASHCODE_FIX = """
# cadquery 2.3 ↔ cadquery-ocp 7.9 compat: OCP removed TopoDS_*.HashCode but
# cq's exporter still calls it. Restore as identity-based stub.
from OCP.TopoDS import (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
    TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)
for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
             TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid):
    if not hasattr(_cls, "HashCode"):
        _cls.HashCode = lambda self, ub=2147483647: id(self) % ub
# Stub show_object (defined only inside CQ-editor) so model code with it doesn't crash
def show_object(*a, **k): pass
"""


def _patch_export(code: str, out_step: Path) -> str:
    """Prepend OCP/cq compat shim, then replace last .exportStep("...") to write
    to out_step (append if model didn't include one)."""
    out_lit = str(out_step).replace("\\", "\\\\")
    patched = re.sub(
        r'(\.exportStep\s*\()["\'].*?["\']\s*\)',
        f'.exportStep("{out_lit}")',
        code,
    )
    if ".exportStep" not in patched:
        # The prompt asks for "the final solid in `result`", not for a Workplane:
        # raw Shapes (Solid/Compound/...) have .exportStep directly, only
        # Workplane needs .val() first.
        patched += f'\n(result.val() if hasattr(result, "val") else result).exportStep("{out_lit}")\n'
    return _OCP_HASHCODE_FIX + "\n" + patched


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a timed-out child and everything it spawned."""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def execute_cq_to_step(code: str, step_path: Path, timeout: int = 300) -> None:
    """Execute `code` so `result` is exported to `step_path`. Raises on failure."""
    step_path.parent.mkdir(parents=True, exist_ok=True)
    if step_path.exists():
        step_path.unlink()
    patched = _patch_export(code, step_path)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8", newline="\n"
    ) as f:
        f.write(patched)
        tmp = f.name
    # Its own session, so a timeout can kill the whole process tree. OCC spawns
    # helper threads/processes that inherit the pipes; killing only the direct
    # child leaves them holding stdout open and communicate() then blocks
    # forever -- the timeout above would never actually fire.
    popen_kwargs = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(
            [sys.executable, tmp],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            _kill_tree(proc)
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass  # Orphan holding the pipe: the tree is dead, stop waiting.
            raise RuntimeError(f"timeout after {timeout}s") from e
        returncode = proc.returncode
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if returncode != 0:
        err = stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown subprocess error"]
        raise RuntimeError(err[0][:300])
    if not step_path.exists():
        raise RuntimeError("subprocess succeeded but no STEP file written")
