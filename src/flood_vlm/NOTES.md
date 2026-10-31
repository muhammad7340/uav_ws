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
