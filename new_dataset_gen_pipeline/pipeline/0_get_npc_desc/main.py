from typing import List

from prefect import task
import subprocess
import os
from pathlib import Path

# region vars
REPO_ULLAMA_URL = os.getenv('REPO_ULLAMA_URL')
REPO_ULLAMA_PLUGIN_URL = os.getenv('REPO_ULLAMA_PLUGIN_URL')
UE_DIR_PATH = os.getenv('UE_DIR_PATH')
PROJECT_DIR = Path(os.getenv('PROJECT_DIR')).resolve()
PROJECT_F_NAME = os.getenv('PROJECT_F_NAME')
BRANCH = os.getenv('BRANCH')
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
    if not PROJECT_DIR.exists():
        print("Cloning repository...")
        run(["git", "clone", REPO_ULLAMA_URL, str(PROJECT_DIR)])
        run(["git", "clone", REPO_ULLAMA_PLUGIN_URL, str(ULLAMA_PLUGIN_DIR)])
    else:
        print("Repository already exists")


def update_repo():
    run(["git", "fetch", "--all"], cwd=PROJECT_DIR)
    run(["git", "checkout", BRANCH], cwd=PROJECT_DIR)
    run(["git", "pull"], cwd=PROJECT_DIR)


def checkout_commit(git_commit: str):
    run(["git", "checkout", git_commit], cwd=PROJECT_DIR)


def build_unreal_project():
    cmd = [
        f"{UE_DIR_PATH}/Engine/Build/BatchFiles/Build.bat",
        f"{PROJECT_F_NAME}Editor",
        "Win64",
        "Development",
        f"{UPROJECT}"
    ]

    run(cmd)


def extract_npc_from_dataasset(npc_lst: List[str], git_commit: str):
    output_dir = f'input_data/{git_commit}/'

    absolute_dir_path = Path(output_dir).resolve().as_posix()

    os.makedirs(absolute_dir_path, exist_ok=True)

    EXPORT_ARGS = [
        "--output_dir", absolute_dir_path,
    ]

    for npc in npc_lst:
        EXPORT_ARGS.append(f"--npc")
        EXPORT_ARGS.append(npc)

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

@task()
def main(git_commit: str, npc_list: List[str]):
    ensure_repo()
    update_repo()
    checkout_commit(git_commit=git_commit)
    build_unreal_project()
    extract_npc_from_dataasset(npc_lst=npc_list, git_commit=git_commit)


if __name__ == "__main__":
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"
    NPC_LIST = ['trader']
    exit(main(git_commit=COMMIT, npc_list=NPC_LIST))