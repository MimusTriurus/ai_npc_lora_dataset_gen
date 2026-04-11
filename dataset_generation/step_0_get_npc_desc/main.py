from prefect import task
import subprocess
import os
from pathlib import Path

from common.constants import DATA_DIR_NAME

# region vars
REPO_ULLAMA_URL = os.getenv('STEP_0_REPO_ULLAMA_URL')
REPO_ULLAMA_PLUGIN_URL = os.getenv('STEP_0_REPO_ULLAMA_PLUGIN_URL')
UE_DIR_PATH = os.getenv('STEP_0_UE_DIR_PATH')
PROJECT_DIR = os.getenv('STEP_0_PROJECT_DIR')
#PROJECT_DIR = Path(PROJECT_DIR).resolve()
PROJECT_F_NAME = os.getenv('STEP_0_PROJECT_F_NAME')
BRANCH = os.getenv('STEP_0_BRANCH')
# endregion

UPROJECT = f"{PROJECT_DIR}/{PROJECT_F_NAME}.uproject"
ULLAMA_PLUGIN_DIR = Path(f"{PROJECT_DIR}/Plugins/ULlama")

UE_BIN_DIR_PATH = f'{UE_DIR_PATH}/Engine/Binaries'
UE_EDITOR = f"{UE_BIN_DIR_PATH}/Win64/UnrealEditor-Cmd.exe"

EXPORT_SCRIPT: str = Path(f'{os.path.dirname(os.path.abspath(__file__))}/export_npc_dataasset.py').as_posix()

def run(cmd, cwd=None):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo():
    project_dir = Path(PROJECT_DIR).resolve()
    if not project_dir.exists():
        print("Cloning repository...")
        run(["git", "clone", REPO_ULLAMA_URL, str(project_dir)])
        run(["git", "clone", REPO_ULLAMA_PLUGIN_URL, str(ULLAMA_PLUGIN_DIR)])
    else:
        print("Repository already exists")


def update_repo():
    project_dir = Path(PROJECT_DIR).resolve()
    run(["git", "fetch", "--all"], cwd=project_dir)
    run(["git", "checkout", BRANCH], cwd=project_dir)
    run(["git", "pull"], cwd=project_dir)


def checkout_commit(git_commit: str):
    project_dir = Path(PROJECT_DIR).resolve()
    run(["git", "checkout", git_commit], cwd=project_dir)


def build_unreal_project():
    cmd = [
        f"{UE_DIR_PATH}/Engine/Build/BatchFiles/Build.bat",
        f"{PROJECT_F_NAME}Editor",
        "Win64",
        "Development",
        f"{UPROJECT}"
    ]

    run(cmd)


def extract_npc_from_dataasset(npc_name: str, git_commit: str, flow_run_id: str):
    output_dir = f'{DATA_DIR_NAME}/{git_commit[:7]}/'

    absolute_dir_path = Path(output_dir).resolve().as_posix()

    os.makedirs(absolute_dir_path, exist_ok=True)

    EXPORT_ARGS = [
        "--output_dir", absolute_dir_path,
        "--npc", npc_name,
        "--flow_run_id", flow_run_id,
    ]

    script_arg = EXPORT_SCRIPT

    if EXPORT_ARGS:
        script_arg += " " + " ".join(EXPORT_ARGS)

    cmd = [
        UE_EDITOR,
        UPROJECT,
        "-run=pythonscript",
        "-DisableLiveCoding",
        f"-script={script_arg}",
    ]
    run(cmd)

@task(name="step_0_extract_npc_from_ue_dataasset")
def process(git_commit: str, npc_name: str, flow_run_id: str = None):
    ensure_repo()
    update_repo()
    checkout_commit(git_commit=git_commit)
    build_unreal_project()
    extract_npc_from_dataasset(npc_name=npc_name, git_commit=git_commit, flow_run_id=flow_run_id)

if __name__ == "__main__":
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID))