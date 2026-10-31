# FloodScout-VLM — Build Plan for the Coding Agent

> **Project:** Language-guided UAV flood-relief scout with a CPU-only VLM (Qwen3.5-4B via llama.cpp) in ROS 2 Jazzy + Gazebo Harmonic.
> **Repo:** `uav_ws` (github.com/muhammad7340/uav_ws). Put this file at the repo root.
> **How to use:** Tell the agent: *"Read FLOOD_VLM_PLAN.md and do Phase N. Stop when its acceptance checks pass."*

---

## 0. Rules for the agent (read first)

1. Work **one phase at a time**. Finish every acceptance check in a phase before starting the next.
2. **Do not refactor unrelated code.** Changes to existing files are limited to what is listed in each phase.
3. New code goes in a **new package `src/flood_vlm`** (ament_python). Only touch `drone_basic` where this plan says so.
4. Build with `colcon build --symlink-install --packages-select drone_basic flood_vlm` and `source install/setup.bash`.
5. Never delete user files. Commit at the end of each phase: `git commit -m "flood_vlm: phase N - <title>"`.
6. If something in this plan turns out wrong (topic name, frame name, API flag), **check the real system** (`ros2 topic list`, `ros2 run tf2_tools view_frames`, `gz service -l`, `llama-server --help`), fix it, and note the change in `src/flood_vlm/NOTES.md`.
7. Keep the machine responsive: **16 GB RAM, CPU only.** Run RViz only when needed (`rviz:=false`), keep camera rates low, pause the sim while the VLM thinks.

---

## 1. The story (what we are building)

A monsoon flood has hit a town. A responder types:

> *"Find everyone who is stranded, judge who needs help first, and mark a safe spot near each person for the rescue boat."*

The drone, driven by a 4B vision-language model on CPU:

1. **Plans** a survey from the text (validated JSON tool calls).
2. **Surveys** the town from 20 m and **grounds** people, dry areas and roads in the down-camera image, then converts them to world coordinates.
3. **Looks again** from 8 m when unsure, to confirm the person, their posture (sitting/standing/lying) and whether they are on a roof or in water.
4. **Triages**: ranks people by urgency with a short reason.
5. **Marks** a safe, dry spot near each person by hovering there and spawning an orange **rescue beacon** in Gazebo, and publishes a rescue request.
6. **Re-plans when the water rises** mid-mission (low roofs become critical).
7. **Reports**: ranked rescue list, annotated top-down map, metrics.

The evaluation has three parts: closed-loop missions on generated scenes, **FloodSim-VQA** (our auto-generated questions), and zero-shot **FloodNet VQA** (real UAV flood images).

---

## 2. Facts about the existing repo (verified)

| Item | Value |
|---|---|
| Main package | `src/drone_basic` (ament_python, `setup.py`) |
| Launch | `drone_basic/launch/spawn_robot.launch.py` (args: `world`, `rviz`, `x`, `y`, `yaw`) → includes `world.launch.py` |
| World path | `<share>/drone_basic/worlds/<world>`; extra model path `~/gazebo_models` is added to `GZ_SIM_RESOURCE_PATH` |
| Flight node | `drone_basic/drone_basic/drone_way_home.py` (class `GoHome`): FSM on `odometry/filtered`, publishes `drone/cmd_vel`, takes string waypoints `"1"`–`"5"` / `"home"` on `drone/waypoint`, service `go_home`. Publishes **body-frame** velocities. Aborts on any external `cmd_vel`. |
| Flight params today | `takeoff_height 3.0`, `cruise_speed 0.5`, `ascend_speed 0.6`, `descend_speed 0.5`, `position_tolerance 0.3` |
| Down camera | `/camera_down/image`, `/camera_down/camera_info` (already bridged), 640×480, HFOV 1.396 rad (80°), **far clip 15 m, update_rate 1 Hz**, frame `camera_down_link_optical`, mounted at rpy (0, 1.5707, 0) |
| Other sensors | front RGB-D + wide-angle cameras at 20 Hz (CPU heavy) |
| Ground truth odom | `drone/odom_ground_truth` (bridged) |
| Spherical coords | `home.sdf` has `<spherical_coordinates>` (lat 47.478950, lon 19.057785), needed by navsat/EKF. **Copy it into new worlds.** |
| World plugins | copy the `<plugin>` block (Physics, UserCommands, SceneBroadcaster, Sensors with ogre2, IMU, NavSat) from `worlds/home.sdf` |

---

## 3. Resources (download list)

### Model + runtime
| Item | Where | Size |
|---|---|---|
| `Qwen3.5-4B-Q4_K_M.gguf` | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF | 2.74 GB |
| `mmproj-F16.gguf` (vision encoder) | same repo | 672 MB |
| `Qwen3.5-4B-Q8_0.gguf` (ablation only, Phase 4) | same repo | 4.48 GB |
| llama.cpp CPU build (`llama-server`) | https://github.com/ggml-org/llama.cpp/releases → `*-bin-ubuntu-x64.zip` | small |

Store models in `~/models/qwen35/`. **Do not commit models** (add to `.gitignore`).

### Gazebo Fuel models (pre-download with `gz fuel download -u <url>`)
Base URL: `https://fuel.gazebosim.org/1.0/OpenRobotics/models/<Name>`

| Use | Model name | Size | License |
|---|---|---|---|
| Person sitting on roof | `Rescue Randy Sitting` | 2.9 MB | CC0 |
| Person lying (injured) | `Rescue Randy` | 10.7 MB | CC-BY 4.0 |
| Person standing | `Male visitor` | 0.6 MB | CC0 |
| Cars (half submerged) | `Hatchback`, `SUV`, `Pickup` | 0.4–1.9 MB | CC0 |
| Staging area | `Ambulance`, `RC Boat` (CC-BY 4.0) | 0.5 / 3.6 MB | |
| Trees | `Oak tree`, `Pine Tree` | 0.6–0.9 MB | CC0 |
| Landmarks | `Water tower`, `Gas Station` | 1–1.7 MB | CC0 |
| GUI-only waves (optional) | `Coast Water` (60×60 m mesh) | 2 MB | CC-BY 4.0 |

**Do not use** `Standing person`, `Casual female`, `Lake House` (25–230 MB, slow), or the Fuel houses (pitched roofs; people cannot sit on them convincingly).
**Do not use Swishy's water**: it is hidden from cameras (`visibility_flags 4`) and needs a custom plugin.

### Datasets
| Dataset | Use | Where |
|---|---|---|
| FloodNet Track 2 (VQA), 4,511 Q&A on real Hurricane Harvey UAV images. Types: simple counting, complex counting, condition recognition, yes/no. CC BY-SA 4.0 | Real-image zero-shot eval | https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021 (Google Drive link in README) |
| Quick mirror (334 MB, webdataset tars) | Faster download | https://huggingface.co/datasets/ljx620/FloodNet-VQA |
| FloodSim-VQA (ours) | Generated in Phase 4 | from scene ground truth |

---

## 4. Key constants (put in `src/flood_vlm/config/mission.yaml`)

```yaml
world_name: flood_town            # <world name="..."> in generated SDF
area: {xmin: -30, xmax: 30, ymin: -30, ymax: 30}
home: {x: 0.0, y: -26.0}          # staging area (dry mound, ambulance, boat). Drone spawns here.
survey_alt: 20.0                  # one frame covers ~34 x 25 m
look_alt: 8.0
cruise_speed: 2.0
ascend_speed: 1.5
descend_speed: 1.2
settle_s: 1.5                     # wait after arrival before grabbing a frame
water: {start: 0.8, steps: [1.6, 2.4], step_every_s: 120}   # sim seconds
lookagain_conf: 0.6
merge_radius_m: 2.0
beacon_min_dist_from_person_m: 1.5
llm: {url: "http://127.0.0.1:8080/v1/chat/completions", max_tokens_fast: 384, max_tokens_think: 1024, temperature: 0.2}
```

> Spawn the drone at `home` (launch args `x:=0.0 y:=-26.0`). Ground truth is in world frame. EKF `odom` starts at the spawn point, so **world = odom + home offset**. Handle this in one function (`geometry.odom_to_world`) and test it.

---

## PHASE 1 — Flood world, sim plumbing, VLM server  *(~1.5 h)*

**Goal:** a generated flooded town loads, the drone flies to any 3D pose on command, water can rise, beacons can be spawned, and the VLM answers a question about a sim image.

### 1.1 VLM server (do first, it downloads while you code)
- `scripts/setup_vlm.sh`: download llama.cpp release + the two GGUF files into `~/models/qwen35/`.
- `scripts/run_vlm.sh`:
  ```bash
  llama-server -m ~/models/qwen35/Qwen3.5-4B-Q4_K_M.gguf \
    --mmproj ~/models/qwen35/mmproj-F16.gguf \
    -c 8192 -t $(( $(nproc) / 2 )) --host 127.0.0.1 --port 8080 --jinja
  ```
  Thinking is toggled **per request** with `"chat_template_kwargs": {"enable_thinking": false|true}`. If the server ignores it, use the launch flag `--chat-template-kwargs '{"enable_thinking":false}'` and run planning with a second request style. Note the result in `NOTES.md`.

### 1.2 Scene generator — `flood_vlm/scene_gen.py`
CLI: `ros2 run flood_vlm scene_gen --seed 7 --out <drone_basic>/worlds/flood_town_7.sdf` → also writes `flood_town_7.gt.json` next to it.

Builds (all procedural boxes unless noted):
- **Ground:** 80×80 m plane, brown-green.
- **Flat-roof houses (8–12)**, Pakistani style: footprint 6–9 m, 1 storey (3 m) or 2 storeys (6 m), some on a 0.6–1.0 m plinth. Roof parapet 0.4 m high. A black cylinder **water tank** on some roofs. Colours: off-white, beige, pale yellow, light pink, grey. Place on a loose street grid with ≥4 m gaps.
- **Roads:** grey strips along the grid. Mark 1–2 segments as **raised embankments** (top 0.3 m above the *final* water level, so always dry); the rest are at ground level (submerged).
- **Dry mounds (2–3):** brown-green boxes 6×6 m, top above the final water level. One is the **staging area** at `home` with `Ambulance` and `RC Boat`.
- **Flood water:** model `flood_water`, a single **visual-only** box 80×80×0.05 m, muddy brown `(0.45, 0.36, 0.22)`, transparency 0.15, **no collision**, top at `water.start`. Not static is fine; it must accept `set_pose`.
- **People (6–9):** mix of
  - `Rescue Randy Sitting` on roofs (on roof top + parapet inner area),
  - `Male visitor` standing on roofs or **standing in water** next to a car/house (water up to waist),
  - `Rescue Randy` **lying on a roof** (1–2, injured).
- **Cars (3–5)** half submerged on roads, **trees (6–10)**, 1 `Water tower`, 1 `Gas Station` as landmarks.
- Copy `<spherical_coordinates>` and the system plugins from `worlds/home.sdf`. World name `flood_town`.
- Keep it light: no shadows on boxes, ≤ ~60 models.

**Ground-truth JSON (`*.gt.json`):**
```json
{"seed":7,"water":{"start":0.8,"steps":[1.6,2.4]},
 "houses":[{"id":"H1","x":..,"y":..,"yaw":..,"sx":..,"sy":..,"roof_z":..,"plinth_z":..}],
 "people":[{"id":"P1","x":..,"y":..,"z":..,"model":"rescue_randy_sitting",
            "support":"roof|water|mound","house":"H3|null","posture":"sitting|standing|lying"}],
 "dry_zones":[{"id":"D1","type":"mound|road|roof","polygon":[[x,y],...],"top_z":..}],
 "roads":[{"id":"R1","polygon":[[x,y],...],"top_z":..}],
 "cars":[...],"home":{"x":0,"y":-26}}
```

**Urgency rule (ground truth, `flood_vlm/gt.py`):** for water level `w`:
- `support=="water"` → **critical**
- `posture=="lying"` → **critical**
- roof person with `roof_z - w < 1.0` → **high**, else **medium**
- GT ranking = sort by (level, roof_z - w). Recompute whenever `w` changes.

**Valid beacon spot (ground truth):** inside a `dry_zone` (or the same roof) whose `top_z > w + 0.2`, distance to its person ≥ 1.5 m and ≤ 15 m.

### 1.3 Drone changes in `drone_basic` (minimal)
1. `urdf/drone_custom.gazebo`, `camera_down`: `<far>60</far>`, `<update_rate>2</update_rate>`. RGB-D + wide-angle cameras: `<update_rate>5</update_rate>`.
2. `drone_way_home.py`: add a **3D goal input**.
   - Subscribe `drone/goal_pose` (`geometry_msgs/PoseStamped`, frame `odom`).
   - New states: `GOAL_Z` (climb or descend to goal z first, if goal z > current z, otherwise after XY), `GOAL_XY`, `GOAL_HOLD`.
   - A new goal **pre-empts** any current goal state (not a manual cmd_vel).
   - Publish `drone/state` (`std_msgs/String`, the FSM state) at 5 Hz, and `drone/goal_reached` (`std_msgs/Bool`, true once on reaching). Keep `HOLD` publishing zero twist so it hovers.
   - Keep all existing behaviour (waypoints, home, override) unchanged.
3. Launch: in `spawn_robot.launch.py`, make the waypoint node read `cruise_speed`, `ascend_speed`, `descend_speed` from launch args (defaults stay the same). Our launch sets 2.0 / 1.5 / 1.2.

### 1.4 Gazebo tools — `flood_vlm/gz_tools.py`
Thin wrappers over `gz service` (subprocess, 3 s timeout, return bool):
- `pause(True|False)` → `/world/flood_town/control` (`gz.msgs.WorldControl`)
- `set_water(z)` → `/world/flood_town/set_pose` on `flood_water`
- `spawn_beacon(id, x, y, z)` → `/world/flood_town/create` with an inline SDF: orange cylinder (r 0.3, h 1.2) + emissive sphere on top, visual only, name `beacon_<id>`.
- `water_controller` node: raises water by the `water.steps` schedule (sim time), publishes `/flood/water_level` (`std_msgs/Float32`, latched) and logs each change.

### 1.5 Launch — `flood_vlm/launch/flood_world.launch.py`
Includes `drone_basic/spawn_robot.launch.py` with `world:=flood_town_<seed>.sdf rviz:=false x:=0 y:=-26`, plus `water_controller`.

### ✅ Phase 1 acceptance
- [ ] `scene_gen --seed 7` makes a world that loads in < 60 s with no missing models. Screenshot from above saved to `docs/flood/scene_seed7.png`.
- [ ] `ros2 topic pub --once drone/goal_pose ...` to (10, 5, 20) → drone climbs, flies, hovers; `drone/goal_reached` goes true. Error vs `drone/odom_ground_truth` < 0.5 m.
- [ ] `/camera_down/image` at 20 m shows roofs, people and brown water (save `docs/flood/down_20m.png`).
- [ ] `set_water(1.6)` visibly raises the water; `spawn_beacon` shows a beacon.
- [ ] `curl` to llama-server with `down_20m.png` and "Describe the scene and point to every person (JSON, 0-1000 coords)" returns sensible JSON. Record latency in `NOTES.md`.

---

## PHASE 2 — VLM perception core  *(~1.5 h)*

**Goal:** from one camera frame + drone pose, get a list of grounded objects in **world coordinates** with confidence, merged into an object memory.

### 2.1 `flood_vlm/vlm_client.py`
- `ask(image: np.ndarray|None, prompt: str, schema: dict, think: bool, max_tokens: int) -> (dict, timings)`.
- Sends OpenAI-style chat with the image as base64 JPEG (quality 90, keep 640×480).
- Uses `response_format: {"type":"json_schema","json_schema":{"name":"out","schema":schema}}` so output is always valid JSON. If parsing still fails: retry once with `temperature 0`, then return `{"error":...}`.
- Returns llama-server `timings` (prompt_n, prompt_ms, predicted_n, predicted_ms) and wall time. **Log every call** to `runs/<run_id>/vlm_calls.jsonl` (prompt id, image path, output, timings).

### 2.2 Prompts + schemas — `flood_vlm/prompts.py`, `flood_vlm/schemas.py`
Coordinates are **0–1000 relative** (Qwen convention). Keep prompts short and factual.

**P_SURVEY** (think off):
> "Aerial top-down photo of a flooded town. Brown is flood water. Find every PERSON. For each give bbox_2d [x1,y1,x2,y2] in 0-1000, where they are (on_roof / in_water / on_dry_ground), posture (sitting / standing / lying / unsure), and confidence 0-1. Also give bboxes of DRY areas (dry roofs, mounds, dry raised road) and say for each road you see whether it is passable. Return JSON only."

Schema `SURVEY`: `{people:[{bbox_2d:[4 ints], location:enum, posture:enum, confidence:number}], dry_areas:[{bbox_2d, kind:enum[roof,mound,road]}], roads:[{bbox_2d, passable:bool}], flooded_fraction:number}`

**P_LOOK** (think off, image from 8 m centred on the target):
> "Close aerial view. Is there a person near the centre? Give bbox_2d, location, posture, whether the surface they are on is dry, and the best SAFE point to place a rescue beacon: dry, flat, at least ~1.5 m from the person, reachable by boat (next to water). Give it as point_2d [x,y] in 0-1000 and one short reason."

Schema `LOOK`: `{person_present:bool, bbox_2d, location, posture, surface_dry:bool, beacon_point_2d:[2 ints], reason:string, confidence:number}`

### 2.3 Pixel → world — `flood_vlm/geometry.py`
- Use `camera_info` K and TF `odom → camera_down_link_optical` at the frame timestamp.
- Ray from camera centre through pixel `(u,v)` (convert 0–1000 → pixels: `u = x/1000*W`, `v = y/1000*H`), intersect with horizontal plane `z = z_plane`.
- `z_plane`: water level for `in_water`, `roof_z` estimate for `on_roof` (use the nearest GT-free estimate: start with `water_level + 2.0`; the look-again step re-centres directly above, which removes most parallax), `top of mound` for mounds.
- `odom_to_world()` / `world_to_odom()` using the home offset.
- **Unit test** (`test/test_geometry.py`): synthetic camera at 20 m looking down; centre pixel → point under drone; corner pixel → expected offset from HFOV.
- **Calibration check in sim:** hover at 20 m over a known person from GT, ground them, report error. If X/Y are swapped or mirrored, fix the axis convention here only. Target error < 1.5 m at 20 m, < 0.5 m after look-again.

### 2.4 Object memory — `flood_vlm/memory.py`
- `ObjectMemory`: list of `Person{id, xyz_world, location, posture, conf, n_obs, crops[], beacon_xy, status}` and `DryZone{...}`.
- Merge a new detection into an existing person if within `merge_radius_m`; average position weighted by confidence (look-again observations get weight ×3).
- `to_text()` → compact JSON for the triage prompt.
- Save crops (from bbox) to `runs/<id>/crops/` for the report.

### ✅ Phase 2 acceptance
- [ ] `scripts/try_survey.py`: drone at 20 m over a quadrant → prints grounded people in world coords and matches them to GT (Hungarian, 3 m gate). Shows recall/precision for that frame.
- [ ] Mean localization error documented in `NOTES.md` (survey vs look-again).
- [ ] All VLM outputs parse (0 JSON failures over 10 calls).

---

## PHASE 3 — Mission agent (the demo)  *(~2 h)*

**Goal:** one command runs the whole story end-to-end and produces a report, map and video.

### 3.1 `flood_vlm/mission_node.py` (rclpy, single-threaded executor + a worker thread for VLM calls)
Tools (each is a Python method; the planner may only call these):

| Tool | Args | Does |
|---|---|---|
| `takeoff` | `alt` | goal_pose at home, z=alt |
| `survey` | `alt`, `cells` | fly a lawnmower over the area; 4 frames at 20 m cover 60×60 m (cells = list of [x,y] centres) |
| `look_again` | `person_id` | fly to `look_alt` above the estimate, centre, run P_LOOK |
| `triage` | – | text-only reasoning call (think **on**) over memory + water level → ranked list with reasons |
| `mark_beacon` | `person_id` | fly to beacon point at 6 m, hover 3 s, `spawn_beacon`, publish request |
| `report` | – | write report + map |
| `return_home` | – | goal_pose home, then land |

**Flow:**
1. **Plan** (think on): input = mission text + area + tools + rules → `PLAN` schema `{steps:[{tool, args}]}`.
   **Validator** (`flood_vlm/validator.py`): tools must exist, args in range (alt 5–25 m, xy inside area), must start with `takeoff` and end with `return_home`. If invalid → feed errors back once; if still invalid → use the default plan. Log validity.
2. **Survey**: at each cell, wait for `goal_reached` + `settle_s`, grab frame, **pause sim**, P_SURVEY, **unpause**, project, merge into memory.
3. **Look again**: for every person with `conf < lookagain_conf` or `posture == unsure` or no beacon point yet → `look_again`. (Ablation flag `--no-lookagain`.)
4. **Triage** (think on): returns `{ranking:[{person_id, level: critical|high|medium, reason}]}`.
5. **Mark** beacons in ranking order. Beacon point = the one from P_LOOK, projected to world. If it fails the dry check *we can measure ourselves* (inside water, i.e. VLM said `surface_dry=false`), ask P_LOOK once more.
6. **Water rises** (`/flood/water_level` callback): set a flag; at the next safe point (between tools) **re-run triage** and re-order the remaining beacons. Log "re-plan because water rose to X m; P4 moved up: roof margin 0.6 m".
7. **Report + return home.**

**Sim pause rule:** pause only while waiting for the VLM, never while the drone is in a goal state that needs physics to finish. Measure "VLM time" and "flight time" separately.

**Rescue request message:** publish JSON string on `/flood/rescue_requests` (`std_msgs/String`): `{person_id, world_xy, level, posture, location, beacon_xy, reason, t}`. Also an RViz `MarkerArray` on `/flood/markers` (people coloured by level, beacons, drone path).

### 3.2 Report — `flood_vlm/report.py`
Writes to `runs/<run_id>/`:
- `report.md`: mission text, plan (+ valid?), ranked rescue list with reasons and crops, re-plan events, timings.
- `map.png` (matplotlib, top-down, world frame): houses, dry zones, water level, GT people (hollow), detected people (filled, coloured by level), beacons (orange stars), drone path, numbers = rescue order.
- `metrics.json`: see Phase 4 definitions (scored automatically against `*.gt.json`).

### 3.3 Launch — `flood_vlm/launch/flood_mission.launch.py`
Args: `seed`, `mission` (text), `lookagain` (bool). Starts the world (Phase 1 launch), waits for odometry, then starts `mission_node`. llama-server is started separately (`scripts/run_vlm.sh`).

### 3.4 Demo recording
- `scripts/record_demo.sh`: starts Gazebo GUI follow-cam on the drone, `ros2 bag record` of `/camera_down/image`, `/flood/*`, `/drone/odom_ground_truth`, and an `ffmpeg` screen capture (`-preset veryfast`). Sim pauses during VLM calls → cut those in editing or show a "VLM thinking…" overlay with the reasoning text.

### ✅ Phase 3 acceptance
- [ ] `ros2 launch flood_vlm flood_mission.launch.py seed:=7 mission:="Find everyone who is stranded, judge who needs help first, and mark a safe spot near each person for the rescue boat."` completes without manual help.
- [ ] Water rises at least once during the run and the log shows a re-plan.
- [ ] Beacons appear in Gazebo; `report.md`, `map.png`, `metrics.json` exist.
- [ ] Peak RAM < 13 GB (`/usr/bin/time -v` or `psrecord`), noted in `NOTES.md`.

---

## PHASE 4 — Datasets and evaluation  *(~2 h, can run overnight)*

**Goal:** numbers that make this research, not a demo.

### 4.1 FloodSim-VQA generator — `flood_vlm/eval/floodsim_vqa_gen.py`
1. For seeds `1..10`: load world, fly a 3×3 grid at 12 m and 20 m (2 water levels each) and save frames + exact camera pose (from `drone/odom_ground_truth`) → ~200 frames in `data/floodsim/`.
2. Project GT into each frame (same camera model as `geometry.py`) and generate questions with exact answers, **same four types as FloodNet + two of ours**:

| Type | Template example | Answer |
|---|---|---|
| Simple counting | "How many people are visible?" | int |
| Complex counting | "How many people are standing in flood water?" / "How many houses have water above the ground floor?" | int |
| Condition recognition | "Is the road in the centre of the image passable or flooded?" | label |
| Yes/No | "Is the person nearest the centre on a dry roof?" | yes/no |
| **Grounding (ours)** | "Point to the person lying down." | correct if point inside projected bbox |
| **Priority (ours)** | "Two people are marked A and B (boxes drawn). Who needs help first?" | A/B from GT urgency |

Target ~1,200 questions. Save `data/floodsim/questions.jsonl`. Hold out seeds 9–10 as test.

### 4.2 Runners
- `eval/run_vqa.py --dataset floodsim|floodnet --n 300 --think off --quant q4`: stratified sample per type, logs answers + timings. Metrics: accuracy per type; counting = exact and ±1.
- `eval/floodnet_prep.py`: load FloodNet Track 2 test split (official or HF mirror), map to the same JSONL format.
- `eval/mission_eval.py --seeds 1..10 --lookagain on|off`: runs full missions headless and aggregates `metrics.json`.

### 4.3 Mission metrics (computed in `flood_vlm/scoring.py` against GT)
| Metric | Definition |
|---|---|
| Plan validity | % plans passing the validator on first try |
| Person recall / precision | Hungarian match, 3 m gate |
| Localization error | mean metres, survey vs after look-again |
| Posture / location accuracy | vs GT |
| Triage quality | Kendall's tau vs GT ranking; % critical people ranked in top-k |
| Safe beacon rate | % beacons satisfying the GT valid-spot rule at the time placed |
| Re-plan correctness | after a water rise, did newly-critical people move up? |
| Cost | VLM seconds, flight seconds, #VLM calls, tokens, peak RAM |

### 4.4 Ablations and CPU profiling (the hardware angle)
Run on FloodSim test (100 q) + 3 missions:
- Q4_K_M vs Q8_0; image long side 448 vs 640; think on vs off for triage; look-again on vs off.
- From llama-server `timings`: split **vision encode + prompt prefill (prompt_ms)** vs **decode (predicted_ms)**; tokens per image; RAM.
- Output `results/ablation.csv` and plots: accuracy vs latency (Pareto), latency breakdown bar chart.

### ✅ Phase 4 acceptance
- [ ] `data/floodsim/questions.jsonl` with ≥ 1,000 questions, answers verified on 20 random samples by eye (drawn overlays saved).
- [ ] FloodNet subset (≥ 300 q) and FloodSim test results in `results/vqa.csv`.
- [ ] 10-seed mission results with look-again on/off in `results/missions.csv`.
- [ ] Ablation table + 2 plots in `results/`.

---

## PHASE 5 — Package the story  *(~1 h)*

1. **README section** "FloodScout-VLM" in `uav_ws/README.md`: one-paragraph story, architecture diagram (Mermaid), how to run (3 commands), results table, limitations.
2. **Figures** in `docs/flood/`: scene screenshot, down-camera frame with grounded boxes, `map.png` from the best run, latency breakdown plot.
3. **Results table** (fill from `results/`): FloodNet accuracy per type, FloodSim accuracy per type, mission metrics with/without look-again, CPU latency and RAM.
4. **90-second video**: mission text → plan → survey frames with boxes → look-again close-up → water rises → re-plan message → beacons → report map.
5. **Limitations + next steps** (honest): simulated victims, single drone, no real payload; next: real FloodNet-style flights, onboard Jetson/NPU or custom accelerator, multi-drone coverage.
6. Final commit + tag `v0.1-floodscout`.

### ✅ Phase 5 acceptance
- [ ] A new person can clone, follow the README, and run one mission.
- [ ] README results table has real numbers from `results/`.

---

## File map (new)

```
src/flood_vlm/
  package.xml  setup.py  setup.cfg  NOTES.md
  config/mission.yaml
  launch/flood_world.launch.py  launch/flood_mission.launch.py
  flood_vlm/
    scene_gen.py  gt.py  gz_tools.py  water_controller.py
    vlm_client.py  prompts.py  schemas.py  validator.py
    geometry.py  memory.py  mission_node.py  report.py  scoring.py
    eval/floodsim_vqa_gen.py  eval/floodnet_prep.py  eval/run_vqa.py  eval/mission_eval.py
  test/test_geometry.py  test/test_gt.py
scripts/setup_vlm.sh  scripts/run_vlm.sh  scripts/try_survey.py  scripts/record_demo.sh
docs/flood/   data/ (gitignored)   runs/ (gitignored)   results/
```

Changed existing files (only these): `drone_basic/urdf/drone_custom.gazebo`, `drone_basic/drone_basic/drone_way_home.py`, `drone_basic/launch/spawn_robot.launch.py`, `.gitignore`, `README.md`.

## Known risks and fallbacks

| Risk | Fallback |
|---|---|
| VLM too slow per frame (> 20 s) | image long side 448; Qwen3.5-2B for survey, 4B only for plan/triage |
| `json_schema` not honoured | GBNF grammar via `"grammar"` field, or strict regex extraction + retry |
| Thinking toggle per request ignored | two llama-server instances is too much RAM; instead keep thinking off and add "Think step by step in the `reason` field" for triage |
| Ray-plane projection off on roofs | rely on look-again re-centring (target near image centre → small parallax) |
| `set_pose` refused on water | delete + re-create `flood_water` at the new height via `create`/`remove` services |
| Fuel download slow | pre-download in Phase 1; if a model fails, swap for a procedural box person (cylinder + sphere) |
| Gazebo + VLM stall the PC | headless `gz sim -s` for eval runs; pause sim during VLM calls; no RViz |
