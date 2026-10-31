import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import  LaunchConfiguration, PathJoinSubstitution, PythonExpression, TextSubstitution


def generate_launch_description():

    world_arg = DeclareLaunchArgument(
        'world', default_value='world.sdf',
        description='Name of the Gazebo world file to load'
    )

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run only the Gazebo server with headless rendering (no GUI)'
    )

    pkg_drone_basic = get_package_share_directory('drone_basic')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Add your own gazebo library path here
    gazebo_models_path = os.path.expanduser("~/gazebo_models")
    os.environ["GZ_SIM_RESOURCE_PATH"] = os.environ.get("GZ_SIM_RESOURCE_PATH", "") + os.pathsep + gazebo_models_path


    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': [PathJoinSubstitution([
            pkg_drone_basic,
            'worlds',
            LaunchConfiguration('world')
        ]),
        TextSubstitution(text=' -r -v -v1'),
        PythonExpression(["' -s --headless-rendering' if '", LaunchConfiguration('headless'), "' == 'true' else ''"])],
        #TextSubstitution(text=' -r -v -v1 --render-engine ogre --render-engine-gui-api-backend opengl')],
        'on_exit_shutdown': 'true'}.items()
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(headless_arg)
    launchDescriptionObject.add_action(gazebo_launch)

    return launchDescriptionObject