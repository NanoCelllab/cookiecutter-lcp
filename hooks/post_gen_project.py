#!/usr/bin/env python3
from pathlib import Path
import os, sys, subprocess

INCLUDE_PIPELINES = "{{ cookiecutter.include_example_pipelines }}".strip().lower()
INCLUDE_NOTEBOOKS = "{{ cookiecutter.include_example_notebooks }}".strip().lower()
INCLUDE_MODELS    = "{{ cookiecutter.include_example_models }}".strip().lower()

def yn(val: str) -> bool:
    return val in ("y", "yes", "true", "1")

def remove_example_files_only(base: Path) -> int:
    removed = 0
    if not base.exists():
        return removed
    for p in base.rglob("*.example"):
        try:
            p.unlink()
            removed += 1
            print(f"[cookiecutter] REMOVED example file: {p}")
        except Exception as e:
            print(f"[cookiecutter] WARN: could not remove {p}: {e}", file=sys.stderr)
    return removed

def conditional_exclusions_keep_dirs(root: Path) -> None:
    if not yn(INCLUDE_PIPELINES):
        removed = remove_example_files_only(root / "workspace" / "pipelines")
        print(f"[cookiecutter] pipelines: removed {removed} example files (kept folder)")
    if not yn(INCLUDE_NOTEBOOKS):
        removed = remove_example_files_only(root / "notebooks")
        print(f"[cookiecutter] notebooks: removed {removed} example files (kept folder)")
    if not yn(INCLUDE_MODELS):
        removed = remove_example_files_only(root / "workspace" / "models")
        print(f"[cookiecutter] models: removed {removed} example files (kept folder)")

def strip_example_suffix(root: Path) -> None:
    if os.environ.get("CC_SKIP_STRIP_EXAMPLE", "0") in ("1","true","True"):
        print("[cookiecutter] Skipping .example stripping (CC_SKIP_STRIP_EXAMPLE=1).")
        return
    suffix = ".example"
    renamed = 0; skipped = 0
    for p in root.rglob("*"):
        if p.is_file() and p.name.endswith(suffix):
            newp = p.with_name(p.name[:-len(suffix)])
            if newp.exists():
                print(f"[cookiecutter] SKIP rename (exists): {p} -> {newp}")
                skipped += 1
                continue
            try:
                p.rename(newp)
                print(f"[cookiecutter] RENAMED: {p} -> {newp}")
                renamed += 1
            except Exception as e:
                print(f"[cookiecutter] ERROR: cannot rename {p}: {e}", file=sys.stderr)
    print(f"[cookiecutter] Strip summary: renamed={renamed} skipped={skipped}")

def try_git_bootstrap(root: Path) -> None:
    def run(cmd): subprocess.run(cmd, check=True, cwd=str(root))
    try:
        run(["git","--version"])
        run(["git","init"])
        run(["git","add","."])
        run(["git","commit","-m","Initial scaffold via cookiecutter-lcp"])
    except Exception as e:
        print(f"⚠️  Warning: git bootstrap skipped: {e}")

def main() -> int:
    root = Path(".").resolve()
    conditional_exclusions_keep_dirs(root)
    strip_example_suffix(root)
    try_git_bootstrap(root)
    print("\n✅ Project scaffold created successfully (folders preserved).")
    print("ℹ️  Next steps:\n   - Create a GitHub repo and push:\n"
          "       git remote add origin <URL>\n"
          "       git branch -M main\n"
          "       git push -u origin main")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
