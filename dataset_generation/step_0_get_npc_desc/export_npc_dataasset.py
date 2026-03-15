import os

import unreal
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--output_dir", type=str)
parser.add_argument("--npc", action="append", help="List of npc to export")
parser.add_argument("--flow_run_id", type=str)
args = parser.parse_args()

asset_path = "/Game/DataAssets/NPCs/NPCsDataRegistry"
data_asset = unreal.load_asset(asset_path)

if data_asset:
    for npc in args.npc:
        json = data_asset.get_npc_data_json('trader')
        out_dir_path = os.path.join(args.output_dir, npc, args.flow_run_id)
        out_f_path = os.path.join(out_dir_path, 'description.json')
        os.makedirs(out_dir_path, exist_ok=True)
        with open(out_f_path, "w", newline="") as f:
            f.write(json)
else:
    unreal.log_error("Can't load npc data asset")
