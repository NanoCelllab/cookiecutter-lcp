# hooks/post_gen_project.py
import pathlib
import subprocess

def run(cmd):
    subprocess.run(cmd, check=True)

base = pathlib.Path(".")

# Best-effort git init (no failure if git missing)
try:
    run(["git", "--version"])
    run(["git", "init"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "Initial scaffold via cookiecutter-lcp"])
except Exception as e:
    print(f"⚠️  Warning: could not run git automatically: {e}")

print("\n✅ Project scaffold created successfully!")
print("ℹ️  Next steps:")
print("   - Create a GitHub repo and push:")
print("       git remote add origin <URL>")
print("       git branch -M main")
print("       git push -u origin main")
