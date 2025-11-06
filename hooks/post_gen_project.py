#!/usr/bin/env python3
# hooks/post_gen_project.py

import os
import sys
from pathlib import Path
import subprocess
from datetime import datetime

INCLUDE_PIPELINES = "{{ cookiecutter.include_example_pipelines }}".strip().lower()
INCLUDE_NOTEBOOKS = "{{ cookiecutter.include_example_notebooks }}".strip().lower()
INCLUDE_MODELS    = "{{ cookiecutter.include_example_models }}".strip().lower()
LICENSE_CHOICE    = "{{ cookiecutter.license }}".strip()
AUTHOR_NAME       = "{{ cookiecutter.author_name }}".strip()
REPO_NAME         = "{{ cookiecutter.repo_name }}".strip()

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
    # pipelines
    if not yn(INCLUDE_PIPELINES):
        removed = remove_example_files_only(root / "workspace" / "pipelines")
        print(f"[cookiecutter] pipelines: removed {removed} example files (kept folder)")

    # notebooks: procurar .example dentro de workspace_dl/**/ (mais realista no seu projeto)
    if not yn(INCLUDE_NOTEBOOKS):
        removed = 0
        base = root / "workspace_dl"
        if base.exists():
            for p in base.rglob("*.example"):
                try:
                    p.unlink()
                    removed += 1
                    print(f"[cookiecutter] REMOVED example file: {p}")
                except Exception as e:
                    print(f"[cookiecutter] WARN: could not remove {p}: {e}", file=sys.stderr)
        print(f"[cookiecutter] notebooks: removed {removed} example files (kept folder)")

    # models
    if not yn(INCLUDE_MODELS):
        removed = remove_example_files_only(root / "workspace" / "models")
        print(f"[cookiecutter] models: removed {removed} example files (kept folder)")

def strip_example_suffix(root: Path) -> None:
    if os.environ.get("CC_SKIP_STRIP_EXAMPLE", "0") in ("1", "true", "True"):
        print("[cookiecutter] Skipping .example stripping (CC_SKIP_STRIP_EXAMPLE=1).")
        return

    suffix = ".example"
    renamed = 0
    skipped = 0
    for p in root.rglob("*"):
        if p.is_file() and p.name.endswith(suffix):
            newp = p.with_name(p.name[: -len(suffix)])
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

def render_license(root: Path) -> None:
    """
    Create LICENSE file based on selection. If None (private), skip.
    We render a simple jinja-like template by replacing tokens manually.
    """
    if LICENSE_CHOICE.lower().startswith("none"):
        print("[cookiecutter] License: None (private). Skipping LICENSE creation.")
        return

    year = str(datetime.now().year)
    lic_map = {
        "MIT": "licenses/MIT.txt.j2",
        "Apache-2.0": "licenses/Apache-2.0.txt.j2",
        "BSD-3-Clause": "licenses/BSD-3-Clause.txt.j2",
    }
    tmpl_rel = lic_map.get(LICENSE_CHOICE)
    if not tmpl_rel:
        print(f"[cookiecutter] Unknown license choice '{LICENSE_CHOICE}'. Skipping.", file=sys.stderr)
        return

    tmpl_path = (Path(".") / tmpl_rel).resolve()
    if not tmpl_path.exists():
        print(f"[cookiecutter] License template not found: {tmpl_path}", file=sys.stderr)
        return

    text = tmpl_path.read_text(encoding="utf-8")
    # Very small token replacement (we already validated English-only inputs)
    text = text.replace("{{ cookiecutter.author_name }}", AUTHOR_NAME)
    text = text.replace("{{ cookiecutter._year }}", year)

    (root / "LICENSE").write_text(text, encoding="utf-8")
    print(f"[cookiecutter] LICENSE created: {LICENSE_CHOICE}")

def badge_for_license(license_choice: str) -> str:
    # Shields.io badges (no repo name needed)
    if license_choice == "MIT":
        return "![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)"
    if license_choice == "Apache-2.0":
        return "![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)"
    if license_choice == "BSD-3-Clause":
        return "![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)"
    return ""

def inject_license_info_into_readme(root: Path) -> None:
    if LICENSE_CHOICE.lower().startswith("none"):
        return

    badge = badge_for_license(LICENSE_CHOICE)
    footer = f"\n---\n**License:** {LICENSE_CHOICE}. See the `LICENSE` file for details.\n"

    for readme_name in ("README_en.md", "README_pt.md", "README.md"):
        p = (root / readme_name)
        if not p.exists():
            continue
        try:
            content = p.read_text(encoding="utf-8")
            # Inject badge right after first header line if possible
            lines = content.splitlines()
            if lines and lines[0].startswith("# "):
                if badge and (badge not in content):
                    lines.insert(1, "")
                    lines.insert(2, badge)
            new_content = "\n".join(lines)
            if footer.strip() not in new_content:
                new_content += footer
            p.write_text(new_content, encoding="utf-8")
            print(f"[cookiecutter] README updated with license info: {p.name}")
        except Exception as e:
            print(f"[cookiecutter] WARN: could not update {p.name}: {e}", file=sys.stderr)

def try_git_bootstrap(root: Path) -> None:
    def run(cmd):
        subprocess.run(cmd, check=True, cwd=str(root))
    try:
        run(["git", "--version"])
        run(["git", "init"])
        run(["git", "add", "."])
        run(["git", "commit", "-m", "Initial scaffold via cookiecutter-lcp"])
    except Exception as e:
        print(f"⚠️  Warning: git bootstrap skipped: {e}")

def main() -> int:
    root = Path(".").resolve()

    # 0) derived values into cookiecutter context (year)
    # We don't re-render jinja; just pass year to license via manual replacement.

    # 1) Respect user choices while keeping folders
    conditional_exclusions_keep_dirs(root)

    # 2) Strip .example suffix from remaining files
    strip_example_suffix(root)

    # 3) Create LICENSE (if applicable)
    render_license(root)

    # 4) Inject badge + footer into READMEs
    inject_license_info_into_readme(root)

    # 5) Best-effort git init/commit
    try_git_bootstrap(root)

    print("\n✅ Project scaffold created successfully.")
    print("ℹ️  Next steps:")
    print("   - Create a GitHub repo and push:")
    print("       git remote add origin <URL>")
    print("       git branch -M main")
    print("       git push -u origin main")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
