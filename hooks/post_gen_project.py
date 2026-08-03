#!/usr/bin/env python3
from pathlib import Path
import subprocess

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
    try_git_bootstrap(root)
    print("\n✅ Project scaffold created successfully (folders preserved).")
    print("ℹ️  Next steps:\n   - Create a GitHub repo and push:\n"
          "       git remote add origin <URL>\n"
          "       git branch -M main\n"
          "       git push -u origin main")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
