"""Full FloodScout mission (FLOOD_VLM_PLAN.md 3.3).

    scripts/run_vlm.sh &          # llama-server, started separately
    ros2 launch flood_vlm flood_mission.launch.py seed:=7 \
        mission:="Find everyone who is stranded, judge who needs help first, and mark a safe spot near each person for the rescue boat."

Starts the flood world (water rises on the mission.yaml schedule, counted from the
moment the mission starts), then the mission node, which waits for odometry itself.
With exit_on_done:=true the whole launch shuts down when the mission ends (eval runs).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEFAULT_MISSION = ('Find everyone who is stranded, judge who needs help first, and mark a safe '
                   'spot near each person for the rescue boat.')


def generate_launch_description():
    pkg = get_package_share_directory('flood_vlm')
    lc = LaunchConfiguration
    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'flood_world.launch.py')),
        launch_arguments={'seed': lc('seed'), 'rviz': lc('rviz'), 'headless': lc('headless'),
                          'water_schedule': lc('water_schedule'), 'water_on_mission': 'true'}.items())
    mission = Node(
        package='flood_vlm', executable='mission_node', name='flood_mission', output='screen',
        parameters=[{
            'use_sim_time': True,
            'seed': ParameterValue(lc('seed'), value_type=int),
            'mission': ParameterValue(lc('mission'), value_type=str),
            'lookagain': ParameterValue(lc('lookagain'), value_type=bool),
            'think_plan': ParameterValue(lc('think_plan'), value_type=bool),
            'think_triage': ParameterValue(lc('think_triage'), value_type=bool),
            'runs_root': lc('runs_root'),
            'run_id': ParameterValue(lc('run_id'), value_type=str),
            'image_long_side': ParameterValue(lc('image_long_side'), value_type=int),
        }])
    shutdown = RegisterEventHandler(
        OnProcessExit(target_action=mission, on_exit=[EmitEvent(event=Shutdown(reason='mission done'))]),
        condition=IfCondition(lc('exit_on_done')))
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='7'),
        DeclareLaunchArgument('mission', default_value=DEFAULT_MISSION),
        DeclareLaunchArgument('lookagain', default_value='true'),
        DeclareLaunchArgument('think_plan', default_value='false'),
        DeclareLaunchArgument('think_triage', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('water_schedule', default_value='true'),
        DeclareLaunchArgument('runs_root', default_value=os.path.join(os.getcwd(), 'runs')),
        DeclareLaunchArgument('run_id', default_value=''),
        DeclareLaunchArgument('image_long_side', default_value='0', description='0 = keep 640x480'),
        DeclareLaunchArgument('exit_on_done', default_value='false'),
        world,
        TimerAction(period=8.0, actions=[mission]),
        shutdown,
    ])
