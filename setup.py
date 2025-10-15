import pathlib, textwrap, os

## RUN THIS SCRIPT AND FOLLOW THE COMMANDS PRINTED IN THE OUTPUT FILE ""
import subprocess, sys, shutil, datetime

# Where to capture the installer output
log_path = project_dir / "install.log"

def run_cmd(label: str, cmd: str):
    """Run a shell command, stream output to console, and tee to the log file."""
    print(f"\n[{label}] → {cmd}")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.datetime.now().isoformat()}] $ {cmd}\n")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
        log.write(f"\n[{label}] exit code: {rc}\n")
        if rc != 0: raise RuntimeError(f"{label} failed with exit code {rc}")

project_dir = pathlib.Path("test").resolve()
if project_dir.exists():
    print(f"Project already exists at: {project_dir}")
else:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "test").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "test" / "__init__.py").write_text("")
    (project_dir / "README.md").write_text("# test\n\nProject created by the workshop tutorial notebook.")
    (project_dir / ".gitignore").write_text(".venv\n__pycache__\n*.pyc\n")
    (project_dir / "pyproject.toml").write_text(textwrap.dedent("""
        [project]
        name = "test"
        version = "0.1.0"
        description = "GCN link prediction demo (PyTorch + PyG)"
        requires-python = ">=3.12"

        dependencies = [
            # We will add heavy deps via explicit uv commands below to ensure correct wheels.
        ]

        [tool.uv]
    """))
    print(f"Created project at: {project_dir}")

print("If running locally, now running:")
cmd_py = "uv python install 3.12"
cmd_venv = "cd test && uv venv --python 3.12 .venv"
cmd_src = "source .venv~/bin/activate"

run_cmd("cmd1 (Python install)", cmd_py)
run_cmd("cmd2 (environment)", cmd_venv)
run_cmd("cmd3 (activate env)", cmd_src)


project_dir = pathlib.Path("test").resolve()
assert project_dir.exists(), "Project folder not found. Run the previous cell first."

TORCH_VERSION = os.environ.get("TORCH_VERSION", "2.4.1")
CUDA_TAG     = os.environ.get("TORCH_CUDA_TAG", "cu124")  # e.g., cu121, cu124
PYG_TAG      = os.environ.get("PYG_TORCH_TAG", f"{TORCH_VERSION}+{CUDA_TAG}")  # e.g., 2.4.1+cu124

print("Planned installs:")
print(f"  torch=={TORCH_VERSION} ({CUDA_TAG})")
print(f"  PyG wheels tag: torch-{PYG_TAG}")

torch_index = f"https://download.pytorch.org/whl/{CUDA_TAG}"
torch_pkgs  = f"torch=={TORCH_VERSION} torchvision torchaudio"
cmd1 = f'cd "{project_dir}" && uv add --index-url {torch_index} {torch_pkgs}'
cmd2 = f'cd "{project_dir}" && uv add scikit-learn matplotlib'
pyg_find_links = f"https://data.pyg.org/whl/torch-{PYG_TAG}.html"
cmd3 = f'cd "{project_dir}" && uv add torch-geometric -f {pyg_find_links}'

print("\nRunning locally ....")
# Execute the three steps
run_cmd("cmd1 (PyTorch stack)", cmd1)
run_cmd("cmd2 (scikit-learn)", cmd2)
run_cmd("cmd3 (PyG wheels)", cmd3)

print(f"\nAll done. Full output was saved to: {log_path}")


print("\nVerifying CUDA/GPU after installation")
