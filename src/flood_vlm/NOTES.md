# flood_vlm — notes and deviations from FLOOD_VLM_PLAN.md

Things checked on the real system that differ from (or add to) the plan.

## Phase 1

### Machine / environment
- **CPU is throttled.** i5-1235U (2 P + 8 E cores) running the `power-saver` profile with
  Turbo Boost off (`/sys/devices/system/cpu/intel_pstate/no_turbo = 1`): every core is capped
  at 1.3 GHz. VLM latency below was measured like that *and* with another Gazebo sim
  (Swishy) running at the same time (load average 20-40). For real numbers:
  `powerprofilesctl set performance` and `echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo`,
  and close other simulations.
- **Isolation.** Other Gazebo/ROS sims on the same machine share topics (`/clock`, `/tf`, ...).
  `source scripts/flood_env.sh` sets `ROS_DOMAIN_ID=42` and `GZ_PARTITION=flood_vlm` and adds the
  flood models to `GZ_SIM_RESOURCE_PATH`. Use it in every terminal.
- **Stopping.** A `ros2 launch` started in the background ignores SIGINT and the gz server can
  be orphaned. `scripts/flood_stop.sh` stops only `flood_town_*` processes. (Swishy's
  `swishy.sh --stop` runs `pkill -f "^gz sim"`, which also kills a running flood sim.)

### VLM server
- llama.cpp releases are now `llama-<tag>-bin-ubuntu-x64.tar.gz` (not `.zip`). `setup_vlm.sh`
  pins `b12017` and extracts to `~/models/qwen35/llama.cpp/`.
- `--jinja` + per-request `"chat_template_kwargs": {"enable_thinking": false}` works: the reply has
  no `reasoning_content`. The server also has `--reasoning on|off|auto` and `--reasoning-budget N`.
- `response_format: {"type": "json_object"}` returns parseable JSON.
- Phase 1 check (`down_20m`-style frame over roof H1 with two people, think off, 384 max tokens):
  both people found, boxes within ~10/1000 of the true positions, scene described as
  "flooded town ... brown floodwater". **Latency 303 s wall** (prompt 382 tok at 3.0 tok/s =
  125 s, decode 171 tok at 1.0 tok/s = 177 s), load average 26, sim paused.
  An earlier call at load ~11 took 111 s (prompt 5.4 tok/s, decode 1.9 tok/s).
  RAM of llama-server: 5.2 GB RSS (Q4_K_M + mmproj F16, ctx 8192, 4 slots).

### Scene
- Fuel `Gas Station` renders **black** in ogre2 (OGRE1 material scripts) and is ~20 m wide:
  replaced by a procedural one (white shop, red canopy, yellow pumps). Gazebo also logs missing
  textures for `Hatchback` wheels, `Ambulance` and `Water tower` thumbnails; they still render.
- `Male visitor` is an **actor**, not a model. Standing people use the `walk.dae` actor from
  `nav_ws2` (copied to `models/flood_people/`) walking in place. **Actor trajectory poses are
  composed with the actor `<pose>`**, so the generator writes `<pose>0 0 0 0 0 0</pose>` and puts the
  absolute pose in the waypoints. `walk.dae`'s origin is ~1 m above the feet.
- Extra set dressing copied in: sandbag walls on the staging mound (`nav_ws2` `sandbags_half.dae`),
  floating logs (Swishy's driftwood, scaled 5.6x) that `water_controller` moves with the water.
  Swishy's water itself is not used (hidden from cameras), as the plan says.
- People "in water" stand on a submerged low wall (feet 0.6-0.9 m) next to a car, so they stay
  visible at the final 2.4 m level (chest-deep) instead of disappearing.
- The light visual showed up in camera images; the sun has `<visualize>false</visualize>`.
- `flood_water` is a **static** model: `/world/flood_town/set_pose` moves it fine (checked 0.8 -> 1.6 m).
- `--overview-cam` adds a fixed 1280x960 top-down camera at 75 m (`/flood/overview/image`, 0.5 Hz)
  used for docs and the report; mission worlds can leave it out.

### Frames
- `drone/odom_ground_truth` is in the **world** frame. EKF `odom` starts at the spawn point
  **including height**: spawn on the staging mound gives `world = odom + (0, -26, 2.83)`.
  So the offset is 3D, not just the home XY. `geometry.OdomWorld` measures it once from
  ground truth vs `odometry/filtered` and falls back to `(home.x, home.y, home.z + spawn)`.
- The drone spawns 0.3 m above the staging mound top (`gt.json` `home.z`). `spawn_robot.launch.py`
  now uses its (previously ignored) `z` argument; its default changed from 0 to 0.5 so the old
  behaviour (spawn at 0.5 m) is kept.

### drone_basic changes
- `drone_way_home.py`: `drone/goal_pose` (odom frame) with `GOAL_Z`/`GOAL_XY`/`GOAL_HOLD`; height and
  hold use a P controller clamped to ascend/descend speed (bang-bang at 1.5 m/s oscillates around
  the 5 cm band). `drone/state` at 5 Hz, `drone/goal_reached` once per goal.
  The manual-override check now compares against the last 10 published twists: under load our own
  echo can arrive after the next tick, which would look like an external `cmd_vel`.
  The node now also gets `use_sim_time` so the loop stops while the sim is paused.
- `world.launch.py` / `spawn_robot.launch.py`: `headless` argument (`gz sim -s --headless-rendering`)
  for eval runs, as the plan's risk table suggests.
- Goal test: world (10, 5, 20) -> hover error vs ground truth **< 0.01 m**, `goal_reached` true
  after ~31 s (climb 17 m at 1.5 m/s, 33 m at 2 m/s).
- `spawn_beacon` sends the SDF inline (`sdf:` field). With `sdf_filename` + deleting the temp file
  the service returned true but nothing appeared: `create` is processed asynchronously.
- Down camera orientation (yaw 0): image up = world +x, image right = world -y.

## Phase 2

### Results (`scripts/try_survey.py`, seed 7, real Qwen3.5-4B Q4_K_M, think off)
| run | cell (world) | GT in frame | survey P / R | survey loc err | look-again loc err | JSON fails |
|---|---|---|---|---|---|---|
| `runs/try_survey_seed7` (ray-plane only) | (-15, -15) | 3 | 1.00 / 1.00 | 0.43 m | 0.41 m (0.61 / 0.29 / 0.32) | 0 / 4 |
| `runs/try_survey_seed7_nw` (+ triangulation, point-guided look) | (-19, 13) | 3 | 1.00 / 1.00 | 0.29 m | **0.10 m** (0.05 / 0.18* / 0.05) | 0 / 4 |

\* the close look answered "no person" for the lying person on the 3.9 m roof, so the survey estimate was kept.
With the two `json_object` calls of Phase 1 that is **0 JSON failures in 10 calls**. `response_format`
`json_schema` is honoured by llama-server (grammar-constrained), no retries were needed.

Latency (machine throttled, see Phase 1): survey 266-431 s (429 prompt tokens; 441-526 generated
tokens at 1.5 tok/s dominate), look-again 127-153 s (~400 prompt, ~130 generated).

### Changes against the plan
- **Survey grid.** The image's short side (25 m at 20 m, ~21 m at roof height) lies along world x
  (the drone faces +x), the long side (34 m / ~29 m) along y. The plan's 4 cells at (+-15, +-15)
  leave a strip around x = 0, y = 0 and the edges uncovered, which the offline test caught (recall
  7/9). Default grid is now 3 x 2: x in {-19, 0, 19}, y in {-13, 13} (2-3 m overlap at roof height).
- **Roof height by two-view triangulation.** The plan projects roof people onto z = water + 2 m and
  that z is what triage would see, so a water rise could never make a roof person "high". Each
  person keeps the camera ray through its bbox centre from every view; after look-again the rays are
  intersected (least squares, needs >= 8 deg between rays) and the torso height minus a posture offset
  (sitting 0.6, standing 1.0, lying 0.15 m) gives `surface_z`. Triage gets `roof_height_est` and
  `height_above_water_est` from that. Measured: 3.0 m roofs within 4 cm in the offline test; in sim
  a lying person on a 3.0 m roof came out at 3.7 m (the bbox centre of a long body shifts with view
  angle), a wading person's support at 0.92 m (true 0.76).
- **Offset look-again.** If every view of a person so far was near-vertical (< 10 deg) the look-again
  hovers 3 m to the side so the new ray is ~30 deg off and the height can still be triangulated.
- **Point-guided P_LOOK.** "near the centre (around point [x, y])": the expected position is projected
  into the close-up. Without it the VLM picks whichever person is most central, which on a roof
  with two people (or after the offset) is often the wrong one.
- **Geometry overrides labels.** A support triangulated less than 0.3 m above the water is treated as
  `in_water`, whatever the VLM said (it called the wading person "on_dry_ground" from 20 m and
  "on_roof" from 8 m).
- Boxes slightly outside 0-1000 are clipped instead of dropped.

### Observations
- Posture from above is the weakest output: the walk.dae actors (standing) were called "lying" 2/3
  times at 8 m; Rescue Randy lying was recognised.
- `geometry.py` was checked against the sim: synthetic camera says image up = world +x, image right =
  world -y; the beacon test in Phase 1 shows the same. TF (EKF) camera position agrees with ground truth
  within ~5 cm while hovering.
- The odom->world offset is measured from ground truth vs EKF at start-up (`DroneIO.wait_ready`); it
  is [0.0, -26.0, 2.83] for seed 7.
