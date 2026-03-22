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


def run(cmd):
    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


for experiment_id in EXPERIMENT_IDS:

    run([
        PYTHON,
        SCRIPT_DIR / "run_session.py",
        "--experiment_id", experiment_id,
        "--session_id", "0",
    ])

print("Done.")