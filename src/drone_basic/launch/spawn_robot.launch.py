import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_drone_basic = get_package_share_directory('drone_basic')

    gazebo_models_path, ignore_last_dir = os.path.split(pkg_drone_basic)
    os.environ["GZ_SIM_RESOURCE_PATH"] = os.environ.get("GZ_SIM_RESOURCE_PATH", "") + os.pathsep + gazebo_models_path

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='rviz2.rviz',
        description='RViz config file'
    )

    world_arg = DeclareLaunchArgument(
        'world', default_value='home.sdf',
        description='Name of the Gazebo world file to load'
    )

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo without its GUI (server + headless rendering)'
    )

    model_arg = DeclareLaunchArgument(
        'model', default_value='drone_custom.urdf',
        description='Name of the URDF description to load'
    )

    x_arg = DeclareLaunchArgument(
        'x', default_value='0.0',
        description='x coordinate of spawned robot'
    )

    y_arg = DeclareLaunchArgument(
        'y', default_value='0.0',
        description='y coordinate of spawned robot'
    )
    z_arg = DeclareLaunchArgument(
        'z', default_value='0.5',
        description='z coordinate of spawned robot'
    )

    cruise_speed_arg = DeclareLaunchArgument(
        'cruise_speed', default_value='0.5',
        description='Horizontal speed of the waypoint / goal flight (m/s)'
    )
    ascend_speed_arg = DeclareLaunchArgument(
        'ascend_speed', default_value='0.6',
        description='Climb speed of the waypoint / goal flight (m/s)'
    )
    descend_speed_arg = DeclareLaunchArgument(
        'descend_speed', default_value='0.5',
        description='Descent speed of the waypoint / goal flight (m/s)'
    )

    yaw_arg = DeclareLaunchArgument(
        'yaw', default_value='0.0',
        description='yaw angle of spawned robot'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time'
    )

    # Define the path to your URDF or Xacro file
    urdf_file_path = PathJoinSubstitution([
        pkg_drone_basic,  # Replace with your package name
        "urdf",
        LaunchConfiguration('model')  # Replace with your URDF or Xacro file
    ])

    gz_bridge_params_path = os.path.join(
        pkg_drone_basic,
        'config',
        'gz_bridge.yaml'
    )

    # Generate path to config file
    interactive_marker_config_file_path = os.path.join(
        pkg_drone_basic,
        'config',
        'linear.yaml'
    )

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_drone_basic, 'launch', 'world.launch.py'),
        ),
        launch_arguments={
        'world': LaunchConfiguration('world'),
        'headless': LaunchConfiguration('headless'),
        }.items()
    )

    # Launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_drone_basic, 'rviz', LaunchConfiguration('rviz_config')])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "drone_custom",
            "-topic", "robot_description",
            "-x", LaunchConfiguration('x'), "-y", LaunchConfiguration('y'), "-z", LaunchConfiguration('z'), "-Y", LaunchConfiguration('yaw')  # Initial spawn position
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # Node to bridge /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "drone/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "drone/odom_ground_truth@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            #"/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/camera_down/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera_down/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",            # additional in case of RGBD Camera
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",# additional in case of RGBD Camera
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': Command(['xacro', ' ', urdf_file_path]),
             'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    trajectory_node = Node(
    package='mogi_trajectory_server',
    executable='mogi_trajectory_server',
    name='mogi_trajectory_server',
    parameters=[{'reference_frame_id': 'odom'},
                {'robot_frame_id': 'drone/base_link'}]
)
    trajectory_ground_truth_topic_node = Node(
        package='mogi_trajectory_server',
        executable='mogi_trajectory_server_topic_based',
        name='mogi_trajectory_server_odom_topic',
        parameters=[{'trajectory_topic': 'trajectory_ground_truth'},
                    {'odometry_topic': 'drone/odom_ground_truth'},
                    {'frame_id': 'odom'},]
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_drone_basic, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
             ]
    )

    navsat_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[
            os.path.join(pkg_drone_basic, 'config', 'navsat_transform.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        remappings=[('imu/data', 'imu'),
                    ('gps/fix', 'navsat'), 
                    ('gps/filtered', 'gps/filtered'),
                    ('odometry/gps', 'odometry/gps'),
                    ('odometry/filtered', 'odometry/filtered')]
    )

    waypoint_node = Node(
        package='drone_basic',
        executable='drone_way_home',
        name='drone_way_home',
        parameters=[{
            'odom_topic': 'odometry/filtered',
            'cmd_vel_topic': 'drone/cmd_vel',
            'waypoint_topic': 'drone/waypoint',
            'takeoff_height': 3.0,
            'cruise_speed': ParameterValue(LaunchConfiguration('cruise_speed'), value_type=float),
            'ascend_speed': ParameterValue(LaunchConfiguration('ascend_speed'), value_type=float),
            'descend_speed': ParameterValue(LaunchConfiguration('descend_speed'), value_type=float),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }
        ],)

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(rviz_config_arg)
    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(model_arg)
    launchDescriptionObject.add_action(headless_arg)
    launchDescriptionObject.add_action(x_arg)
    launchDescriptionObject.add_action(y_arg)
    launchDescriptionObject.add_action(z_arg)
    launchDescriptionObject.add_action(yaw_arg)
    launchDescriptionObject.add_action(cruise_speed_arg)
    launchDescriptionObject.add_action(ascend_speed_arg)
    launchDescriptionObject.add_action(descend_speed_arg)
    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(rviz_node)
    launchDescriptionObject.add_action(spawn_urdf_node)
    launchDescriptionObject.add_action(gz_bridge_node)
    launchDescriptionObject.add_action(robot_state_publisher_node)
    launchDescriptionObject.add_action(trajectory_node)
    launchDescriptionObject.add_action(ekf_node)
    launchDescriptionObject.add_action(navsat_node)
    launchDescriptionObject.add_action(trajectory_ground_truth_topic_node)
    launchDescriptionObject.add_action(waypoint_node)

    return launchDescriptionObject
