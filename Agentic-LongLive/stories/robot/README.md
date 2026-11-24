# Robot episode scripts

Scripts for long-horizon robot videos: one first-frame image of a robot workspace plus eight instructions of 8 s each
(64 s of continuous video). Run them with `scripts/run_robot.py` (see [docs/robotics.md](../../docs/robotics.md)).

## Contents

- `01_*.json` … `30_*.json`: the 30 benchmark episodes of the paper's appendix ("Robot Videos", long-horizon robot
  tasks). Gemini drafted each script from the clip's first frame and original instruction, and the drafts were then
  edited by hand. The first instructions finish the clip's own task. Each script then hides one or two things and brings
  each back one to four instructions later. A head camera turns away and back (20 return moments), or a drawer, fridge,
  box or lid is closed and reopened, or an object is covered and uncovered (21 moments). An instruction that brings
  something back never describes it (for example "pull the top left drawer open again and reach in for what you put
  there"), so only memory can tell the generator what to show.
- `demo_drawer.json`, `demo_card.json`: two short scripts written for the project page, on the first frames of episodes
  05 and 13. They use one simple motion per instruction and name the objects the robot touches.

| Project page example | Script | Policy of the page video |
|---|---|---|
| Desk fan | `15_turn_fan_knob.json` | `--policy v4 --seed 3000` (segment 0 keeps seed 1000) |
| Tube in the drawer | `demo_drawer.json` | `--policy hybrid2` |
| Handing over a card | `demo_card.json` | `--policy hybrid2` |

The page videos were chosen among several rollouts. For the drawer and card examples, groups of later instructions
were re-rolled from a kept earlier part of the same script, and the last two instructions of the drawer video also
pinned close-ups of unmoved tabletop objects from the first frame. A single run of a script therefore gives a video of
the same kind, not the page video itself.

## Fields

```text
name, title, summary
camera        "head camera" | "fixed third-person camera" | "fixed overhead camera"
robot         the robot as it appears in the first frame
setting       the workspace
first_frame   file name of the input image (looked up in --image-root)
source        where the first frame comes from (dataset, split, clip id, frame index, the clip's original instruction,
              viewpoint, SHA-256 of our PNG copy)
segments      [{"instruction": ..., "memory_event": ...}]
```

`memory_event` (`none`, `hide: …`, `return: …`) marks what each instruction hides or brings back. It is meant for
evaluation (finding the return moments) and is never shown to the agent or the generator.

## Input images (not included)

The first frames are not distributed with this repository. They are frame 0 of 30 validation clips of
[RoVid-X](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X) (Deng et al., "Rethinking Video Generation Model for the
Embodied World", ICML 2026), 1280×704 RGB. Download the clips under the dataset's license, export frame 0 of each clip
as a PNG and save it under the name in the table, for example:

```bash
mkdir -p data/robot_first_frames
ffmpeg -i <clip>.mp4 -frames:v 1 data/robot_first_frames/05_open_drawer_ego.png
python scripts/run_robot.py --episode stories/robot/05_open_drawer_ego.json --image-root data/robot_first_frames ...
```

`--image path/to/frame.png` uses a single image instead of `--image-root`. Each script's `source.first_frame_sha256`
is the hash of the PNG we used. A PNG written by another encoder can differ in bytes while showing the same pixels, so
treat a mismatch as a hint to compare the images, not as an error. The two demo scripts use `05_open_drawer_ego.png`
and `13_hand_card_to_person.png`.

| `first_frame` | RoVid-X `val` clip id | Camera | Original instruction of the clip |
|---|---|---|---|
| `01_cut_dough.png` | `094c4709647f2c0bbab4b17690d316da` | head camera | Cut the dough into small pieces with a knife using both hands |
| `02_wash_pan.png` | `bf4e06c43dab91bef98914eeae535816` | head camera | Wash the held black frying pan under the faucet |
| `03_fold_towel.png` | `e6c178ca77296696ae58b3c98fe338fe` | head camera | Fold the cloth from right to left |
| `04_adjust_shirt_hanger.png` | `75a08ba2206bc357a470b007d689eb30` | head camera | Adjust the white t-shirt on the hanger |
| `05_open_drawer_ego.png` | `218e6e266ce0814c9a1770f861850092` | head camera | Pull open the drawer with the left arm |
| `06_open_refrigerator.png` | `04aa58647ea944365b652148ffce0f3f` | head camera | Open the refrigerator door with the left arm |
| `07_seal_food_box.png` | `181775c37a2931e1ffd78e6218639101` | head camera | Press the box lid with both hands to seal the box |
| `08_press_printer_button.png` | `27d76d5d1a19094f9d29f54464d6b625` | head camera | Press the black and white copy button on the printer with the left arm |
| `09_press_disinfection_gun.png` | `97ea0cdcec89d0bfc27877bda0172ee3` | head camera | Press the switch button of the disinfection gun |
| `10_stock_drink_shelf.png` | `a675a25508511a481d0d45086d35e281` | head camera | Place the held iced lemon tea on the shelf at the position with iced lemon tea using right arm |
| `11_tie_rope.png` | `15ca7209c682c9f817c2771fec17731f` | head camera | Tie the rope with both hands |
| `12_receive_tray.png` | `6701126349172c99a237355bb4a5b219` | head camera | Receive the rectangular tray with both arms |
| `13_hand_card_to_person.png` | `ec7e7f229c5abb32ff7796b7d0de8eff` | head camera | pick up card and give it to the person |
| `14_open_coffee_machine.png` | `05f9eebfe930fca38d130ebe0bdb774f` | head camera | Open the lid of the capsule coffee machine with both hands |
| `15_turn_fan_knob.png` | `717bce54f7743f78e73c514d202b4bed` | head camera | Twist the knob on the electric fan with right arm |
| `16_water_plant.png` | `72860ca08ae46927e837b7b4131280db` | fixed third-person camera | pour water into the potted plant |
| `17_pour_from_bottle.png` | `5dd14fbe9d0d22dd2a115367c07ee032` | fixed third-person camera | tilt the bottle to pour liquid |
| `18_wipe_blue_paint.png` | `6a6a183cf2f0eb110e67312de34ac32d` | fixed overhead camera | wipe up blue paint |
| `19_open_wooden_drawer.png` | `4edcd5e3c17b42d618b819e5b31212df` | fixed third-person camera | pull the drawer |
| `20_place_flower_in_vase.png` | `81f23bf651720d8be53f44d69c1aa76b` | fixed third-person camera | move flower to vase |
| `21_place_cup_on_plate.png` | `ad2f16615af8ebfafab6eac1c062aa98` | fixed third-person camera | pick up the green cup and place it on the green plate |
| `22_unscrew_bottle_cap.png` | `218f6a69c496175cf8dcbc18961c24e5` | fixed overhead camera | unscrew the bottle cap |
| `23_move_toothbrush.png` | `d914b8dc62f24e028628e45cb42cbedc` | fixed third-person camera | move the toothbrush |
| `24_open_thermos.png` | `5760f6f2adada1499292fca9212319dc` | fixed third-person camera | open the bottle |
| `25_insert_power_plug.png` | `136e43c1599653127c58282e3eb905e3` | fixed third-person camera | insert white plug into power strip |
| `26_stack_wooden_blocks.png` | `5d581f600f1a34bbc4baa8206dfca54a` | fixed overhead camera | pick up the red hexagon block and stack it on the white cuboid block |
| `27_stack_building_bricks.png` | `8ebeb7473e5ec4ddedc6dec4dd16ce6f` | fixed third-person camera | Stack green blocks on a blue base |
| `28_open_bag_zipper.png` | `eff11421f6c43d0c8cbf839a2ce2f604` | fixed overhead camera | open the zipper |
| `29_open_book.png` | `7b4689112bbce478f587822b22499880` | fixed third-person camera | open book |
| `30_place_hammer_on_basket.png` | `108e395ecb0f971af3c0cebc0cff32d6` | fixed third-person camera | pick up the hammer and place it on the basket |
