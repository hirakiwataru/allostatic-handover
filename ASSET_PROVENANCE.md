# Asset Provenance

This repository does not vendor human-robot-gym assets by default.

`scripts/copy_hrgym_assets.py` can copy selected local files from:

`/mnt/k_iwamoto/sim_data/Projects/human-robot-gym/human_robot_gym/models/assets/human/animations/human-robot-animations/RobotHumanHandover`

to:

`assets/vendor/human_robot_gym/human/animations/human-robot-animations/RobotHumanHandover`

The copied `assets/vendor/human_robot_gym/` tree is ignored by git until the
upstream license and redistribution terms are confirmed. The initial Mjlab
`Mjlab-Allostatic-Handover-Yam` task uses primitive human geometry and does not
require these copied assets. The copy script is provided for the later
full-fidelity RobotHumanHandover migration.

For `Mjlab-Allostatic-Handover-Full`, run:

```bash
python3 scripts/copy_hrgym_assets.py --include-full-handover-assets
```

This copies the local HRGym assets needed for the Yam-based full handover port:

- `human/human.xml`
- `human/meshes/**`
- `textures/skin.png`, `textures/jeans.png`, `textures/green-shirt.png`
- `arenas/table_arena.xml`
- `human/animations/human-robot-animations/RobotHumanHandover/*.pkl`
- `human/animations/human-robot-animations/RobotHumanHandover/*_info.json`

These files remain under `assets/vendor/human_robot_gym/`, which is ignored by
git. Do not commit them until the upstream license and redistribution terms have
been reviewed.
