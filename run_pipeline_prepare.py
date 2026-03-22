from pathlib import Path
import subprocess
import sys
import platform

# 1) put experiment ids here
import yaml

# Load experiment IDs from configs/experiments.yml
experiments_yml_path = Path(__file__).parent / "configs" / "experiments.yml"
with open(experiments_yml_path, "r") as f:
    experiments = yaml.safe_load(f)

EXPERIMENT_IDS = list(experiments.keys())
print(EXPERIMENT_IDS)

# 2) set base path by OS
if platform.system() == "Windows":
    PREPATH = Path("D:/")
else:
    PREPATH = Path("/Users/rezek_zhu")

SCRIPT_DIR = PREPATH / "VLM_Mar18" / "script"
PYTHON = sys.executable

print("Generating design matrix + validating attention + preparing embedding cache")
def run(cmd):
    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


for experiment_id in EXPERIMENT_IDS:
    run([
        PYTHON,
        SCRIPT_DIR / "generate_design_matrix.py",
        "--experiment_id", experiment_id,
    ])

    run([
        PYTHON,
        SCRIPT_DIR / "validate_attention.py",
        "--experiment_id", experiment_id,
    ])

    run([
        PYTHON,
        SCRIPT_DIR / "prepare_embedding_cache.py",
        "--experiment_id", experiment_id,
    ])


print("Done.")