
SYSTEM:
You are generating synthetic player questions for an in-game NPC (shopkeeper/trader). 
Each question must implicitly expect the NPC to perform exactly ONE action from the given action set.
If the action requires parameters, the question must clearly target ONE specific parameter (only one).
Do NOT write the NPC's answer. Only write the player's question and the action metadata.

NPC Role & Style:
- The NPC is a trader/merchant who sells weapons, ammo, upgrades, and healing items.
- Keep questions short, natural, and game-appropriate (1–2 sentences).
- Do NOT include offensive content or meta-references (e.g., "you are an NPC").

Available Actions (with parameters):
- DoNothing: []
- NotEnoughGoldForBuy: []
- SoldOut: []
- ShowAllGoodsForSale: []
- ShowAllWeaponsForSale: []
- ShowAllAmmoForSale: []
- ShowAllUpgradesForWeapons: []
- ShowAllHealingItems: []
- SellAmmo: [pistol, shotgun, rifle, SMG, sniper_rifle, rocket_launcher]
- SellWeapon: [pistol, shotgun, rifle, SMG, sniper_rifle, rocket_launcher]
- SellWeaponUpgrade: [scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer, quick_reload_kit, damage_booster, thermal_scope, stability_mod, barrel_upgrade, trigger_upgrade, camouflage_skin]
- SellHealingItem: [bandage, medkit, antidote, adrenaline_shot, regeneration_serum, painkillers, revive_kit]

Output Requirements:
1) Produce a JSON array. Each element is one generated item.
2) Each item MUST have the following fields:
   - "action": string — one of the actions above (exact spelling).
   - "parameter": string — one selected parameter if the action needs one; otherwise an empty string "".
   - "player_question": string — the player's question to the NPC that implies this action should be performed.
3) Constraints:
   - If the action has no parameters, set "parameter" to "".
   - If the action has parameters, choose exactly ONE parameter from the provided list (no arrays, no multiple params).
   - The "player_question" must naturally imply the selected action and parameter (e.g., sell_weapon → “Sell me the pistol.”).
   - No additional fields beyond "action", "parameter", "player_question".
   - JSON must be valid.

Quantity:
- Generate 20 items.

Examples (illustrative only; do not repeat literally):
- For ShowAllWeaponsForSale: 
  { "action": "ShowAllWeaponsForSale", "parameter": "", "player_question": "What weapons are you selling today?" }
- For SellWeapon (pistol):
  { "action": "SellWeapon", "parameter": "pistol", "player_question": "I need a pistol—can you sell me one?" }
- For SellAmmo (SMG):
  { "action": "SellAmmo", "parameter": "SMG", "player_question": "Got SMG rounds? I'd like a pack." }
- For SellWeaponUpgrade (silencer):
  { "action": "SellWeaponUpgrade", "parameter": "silencer", "player_question": "Do you have a silencer I can buy?" }
- For ShowAllHealingItems:
  { "action": "ShowAllHealingItems", "parameter": "", "player_question": "What healing supplies do you have?" }
- For SoldOut:
  { "action": "SoldOut", "parameter": "", "player_question": "Are you out of stock on rifles today?" }
- For NotEnoughGoldForBuy:
  { "action": "NotEnoughGoldForBuy", "parameter": "", "player_question": "I only have a few coins—can I still get a decent weapon?" }
- For DoNothing:
  { "action": "DoNothing", "parameter": "", "player_question": "Hmm... just browsing for now." }

