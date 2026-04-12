from __future__ import annotations

import compileall
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def compile_path(path: Path) -> bool:
    if path.is_dir():
        return compileall.compile_dir(
            str(path),
            quiet=1,
            force=True,
            maxlevels=10,
        )
    return compileall.compile_file(str(path), quiet=1, force=True)


def main() -> int:
    targets = [
        ROOT / "server.py",
        ROOT / "shaper_core.py",
        ROOT / "primitive_backend.py",
        ROOT / "fill_shaper.py",
        ROOT / "final_shaper.py",
        ROOT / "test_gia_structure.py",
        ROOT / "test_fill_pipeline.py",
        ROOT / "gia",
    ]
    ok = True
    for target in targets:
        ok = compile_path(target) and ok

    # Keep the tracked side-by-side pyc in sync for environments that ship it.
    gia_src = ROOT / "gia" / "json_to_gia.py"
    gia_pyc = ROOT / "gia" / "json_to_gia.pyc"
    if gia_src.exists():
        try:
            py_compile.compile(str(gia_src), cfile=str(gia_pyc), doraise=True)
        except py_compile.PyCompileError:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
