# Demo stories

The eight stories of the project page's long-story wall. Each file has a cast (`setup`) and one prompt per 8 s shot;
`variants.h3` holds the same prompt in MiniMax-H3's audio-video format (used by `configs/longlive_h3.yaml` and
`configs/fasth3.yaml`). Unlike LongStory-Online, the prompts restate the look of the characters, as a user would
when directing a story, and they carry no hidden annotations.

| File | Title | Shots | Story |
| --- | --- | ---: | --- |
| `overclock.json` | Overclock | 40 | A maker builds a graphics card from a bare circuit board, puts it into a water-cooled PC and plays the first game on it at night. |
| `second_skin.json` | Second Skin | 15 | An inventor flies the armored suit he built in his garage, then climbs into a giant suit built around it. |
| `plein_air.json` | Plein Air | 15 | A painter carries one canvas from a mountain lake to a snowy village, adds a piece of every place she visits and hangs the finished painting at home. |
| `spare_parts.json` | Spare Parts | 12 | A tiny brick figure gathers car parts one at a time, then builds a little red car. |
| `biscuit.json` | Biscuit | 15 | A corgi rides his scooter to the market, fills his basket and cooks a whole dinner, one dish at a time. |
| `exposure.json` | Exposure | 15 | A girl photographs a red fox at dawn, develops the film in her darkroom and hangs the print on her wall. |
| `mudlark.json` | Mudlark | 14 | A collie gets soaked at the beach and muddy on the farm, then has a bath and falls asleep by the fire. |
| `keeper.json` | Keeper | 16 | A detective follows a silver key through a rainy night, from an alley to a station locker to a lighthouse at dawn. |

```bash
python scripts/run_story.py --config configs/fasth3.yaml --story stories/demo/biscuit.json output_dir=outputs/biscuit
```

The project page shows one seed per story picked among a few; generation is stochastic, so a new run differs in detail.
