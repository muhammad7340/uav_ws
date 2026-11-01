"""Flooded town + drone + water controller (FLOOD_VLM_PLAN.md 1.5).

    ros2 launch flood_vlm flood_world.launch.py seed:=7
    ros2 launch flood_vlm flood_world.launch.py seed:=7 headless:=true water_schedule:=false

Generate the world first: ros2 run flood_vlm scene_gen --seed 7
"""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from flood_vlm.config import load_config


def _setup(context):
    seed = LaunchConfiguration('seed').perform(context)
    cfg = load_config()
    pkg_drone = get_package_share_directory('drone_basic')
    pkg_flood = get_package_share_directory('flood_vlm')

    # model://flood_sandbags, flood_driftwood, flood_people
    os.environ['GZ_SIM_RESOURCE_PATH'] = (os.environ.get('GZ_SIM_RESOURCE_PATH', '')
                                          + os.pathsep + os.path.join(pkg_flood, 'models'))

    # A leftover gz server (the gz CLI ignores SIGTERM once orphaned) would share the
    # world name and partition: two interleaved clocks and a wrong odom->world offset.
    import subprocess
    left = subprocess.run(['pgrep', '-f', 'gz sim .*flood_tow[n]_'], capture_output=True, text=True).stdout.split()
    if left:
        raise RuntimeError(f'a flood_town Gazebo server is still running (pids {left}): '
                           'run scripts/flood_stop.sh first')

    world = f'flood_town_{seed}.sdf'
    world_path = os.path.join(pkg_drone, 'worlds', world)
    gt_path = os.path.splitext(world_path)[0] + '.gt.json'
    if not os.path.exists(world_path):
        raise RuntimeError(f'{world_path} not found: run `ros2 run flood_vlm scene_gen --seed {seed}` '
                           'and rebuild drone_basic (colcon build --symlink-install)')
    with open(gt_path) as f:
        home = json.load(f)['home']

    drone = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_drone, 'launch', 'spawn_robot.launch.py')),
        launch_arguments={
            'world': world,
            'rviz': LaunchConfiguration('rviz'),
            'headless': LaunchConfiguration('headless'),
            'x': str(home['x']), 'y': str(home['y']), 'z': str(home['z'] + 0.3),
            'cruise_speed': str(cfg['cruise_speed']),
            'ascend_speed': str(cfg['ascend_speed']),
            'descend_speed': str(cfg['descend_speed']),
        }.items())

    water = Node(
        package='flood_vlm', executable='water_controller', name='water_controller',
        output='screen',
        parameters=[{'use_sim_time': True, 'world': cfg['world_name'], 'gt_path': gt_path,
                     'enabled': LaunchConfiguration('water_schedule'),
                     'start_on_mission': LaunchConfiguration('water_on_mission')}])

    # Fixed top-down camera, only present in worlds made with --overview-cam.
    overview_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='overview_bridge',
        arguments=['/flood/overview/image@sensor_msgs/msg/Image[gz.msgs.Image'],
        parameters=[{'use_sim_time': True}])
    return [drone, water, overview_bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='7', description='scene seed (flood_town_<seed>.sdf)'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('headless', default_value='false', description='gz server only'),
        DeclareLaunchArgument('water_schedule', default_value='true',
                              description='raise water on the mission.yaml schedule'),
        DeclareLaunchArgument('water_on_mission', default_value='false',
                              description='start the water schedule when the mission starts'),
        OpaqueFunction(function=_setup),
    ])
