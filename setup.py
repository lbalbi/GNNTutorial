import pathlib, textwrap, os

## RUN THIS SCRIPT AND FOLLOW THE COMMANDS PRINTED IN THE OUTPUT FILE ""

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

print("If running locally, now run:")
print(" uv python install 3.12")
print(" cd test && uv venv --python 3.12 .venv")
print(" ource .venv~/bin/activate")


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
cmd1 = f'cd "{project_dir}" && uv pip install --index-url {torch_index} {torch_pkgs}'
cmd2 = f'cd "{project_dir}" && uv pip install scikit-learn'
pyg_find_links = f"https://data.pyg.org/whl/torch-{PYG_TAG}.html"
cmd3 = f'cd "{project_dir}" && uv pip install torch-geometric -f {pyg_find_links}'

print("\nRunning locally in a terminal:")


print("\nVerify CUDA/GPU after installation:")