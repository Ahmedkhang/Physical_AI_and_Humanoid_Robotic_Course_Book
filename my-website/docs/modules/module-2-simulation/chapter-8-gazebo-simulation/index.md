---
id: chapter-8-gazebo-simulation
title: Chapter 8 - Gazebo Simulation Environment
sidebar_label: Chapter 8
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Link from '@docusaurus/Link';

## Learning Outcomes

After completing this chapter, you will be able to:
1. Install and configure the Gazebo simulation environment
2. Create and customize simulation worlds with objects and environments
3. Spawn and control robots within the Gazebo environment
4. Interface Gazebo with ROS 2 for sensor simulation and control
5. Configure physics properties and parameters for accurate simulation
6. Debug common issues in Gazebo-ROS 2 integration
7. Optimize simulation performance for complex scenarios
8. Understand the differences between simulation and real-world robot behavior

## Gherkin Specifications

### Scenario 1: Gazebo Environment Setup
- **Given** a properly installed ROS 2 and Gazebo environment
- **When** starting Gazebo with a specific world file
- **Then** the simulation environment loads correctly with expected objects

### Scenario 2: Robot Integration
- **Given** a robot model in URDF format
- **When** the robot is spawned in Gazebo
- **Then** it appears correctly and can be controlled via ROS 2

### Scenario 3: Sensor Simulation
- **Given** a robot with simulated sensors in Gazebo
- **When** the simulation is running
- **Then** sensor data is published to ROS 2 topics accurately

### Scenario 4: Physics Configuration
- **Given** a simulation with specific physics requirements
- **When** physics parameters are adjusted
- **Then** the simulation behaves according to physical constraints

### Scenario 5: Performance Optimization
- **Given** a complex simulation scenario
- **When** optimization techniques are applied
- **Then** the simulation runs efficiently without significant slowdown

## Theory & Intuition

Think of Gazebo like a film studio for robotics, where you can create realistic scenes with different environments, objects, and lighting conditions. Just as a film studio allows actors to rehearse scenes safely before real-world filming, Gazebo allows robots to rehearse behaviors and test algorithms in a safe, controlled virtual environment.

In a film studio, set designers create detailed props and environments that look and behave realistically on camera. Similarly, Gazebo creates virtual environments and objects with physical properties that simulate real-world physics. The cameras in the studio correspond to the sensors in Gazebo (cameras, LiDAR, IMU), which capture the simulated environment just as real sensors would.

The key advantage is safety and efficiency - you can test complex behaviors repeatedly without risk of damaging expensive hardware, and you can experiment with different scenarios quickly without physical setup.

## Core Concepts

<Tabs
  defaultValue="diagram"
  values={[
    {label: 'Gazebo Architecture', value: 'diagram'},
    {label: 'Gazebo Components Table', value: 'table'},
  ]}>
  <TabItem value="diagram">

```mermaid
graph TB
    subgraph "Gazebo Simulation"
        A[Gazebo Server]
        B[Gazebo Client (GUI)]
        C[Physics Engine]
        D[Sensor System]
        E[Model Database]
    end
    
    subgraph "ROS 2 Interface"
        F[ROS 2 Bridge]
        G[Sensor Topics]
        H[Control Topics]
    end
    
    subgraph "User Components"
        I[Robot Controller]
        J[Sensing Nodes]
        K[Planning Nodes]
    end
    
    A <--> B
    A --> C
    A --> D
    A --> E
    A <--> F
    F <--> G
    F <--> H
    G --> J
    H <-- I
    J --> K
    
    style A fill:#e1f5fe
    style F fill:#f3e5f5
    style I fill:#e8f5e8
```

  </TabItem>
  <TabItem value="table">

| Component | Purpose | Example Use |
|-----------|---------|-------------|
| World Files | Define environment layout | Room, outdoor terrain |
| Models | 3D objects with properties | Robots, furniture, tools |
| Plugins | Extend simulation capabilities | ROS 2 bridge, controllers |
| Sensors | Simulate real sensors | Camera, LiDAR, IMU |
| Physics Engine | Handle object interactions | Gravity, collisions |

  </TabItem>
</Tabs>

## Hands-On Labs

<Tabs
  defaultValue="lab1"
  values={[
    {label: 'Lab 1: Gazebo Environment Setup', value: 'lab1'},
    {label: 'Lab 2: Robot Spawning and Control', value: 'lab2'},
    {label: 'Lab 3: Sensor Simulation Integration', value: 'lab3'},
  ]}>
  <TabItem value="lab1">

### Lab 1: Gazebo Environment Setup

#### Objective
Install and configure Gazebo, then create and run a basic simulation environment.

#### Required Components
- ROS 2 environment
- Gazebo (typically installed with ROS 2 desktop)
- Text editor
- Terminal access

#### Steps
1. Verify Gazebo installation:
   ```bash
   # Check Gazebo version (should be Fortress or Garden for ROS 2 Humble)
   gazebo --version
   ```

2. If Gazebo is not installed, install it:
   ```bash
   sudo apt update
   sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-plugins ros-humble-gazebo-dev
   ```

3. Check for available Gazebo worlds:
   ```bash
   # Find Gazebo worlds
   find /usr -name "*.world" 2>/dev/null | grep gazebo
   # or
   find /opt/ros -name "*.world" 2>/dev/null
   ```

4. Launch Gazebo with an empty world:
   ```bash
   # Source ROS 2 environment
   source /opt/ros/humble/setup.bash
   
   # Launch Gazebo empty world
   gazebo --verbose
   ```

5. Create a custom world file:
   ```bash
   mkdir -p ~/ros2_ws/gazebo_worlds
   nano ~/ros2_ws/gazebo_worlds/my_room.world
   ```

6. Add the following world definition:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="my_room">
       <!-- Include the default atmosphere -->
       <include>
         <uri>model://sun</uri>
       </include>
       
       <!-- Ground plane -->
       <include>
         <uri>model://ground_plane</uri>
       </include>
       
       <!-- A simple room with walls -->
       <model name="room_wall_1">
         <pose>0 5 1 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>10 0.2 2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>10 0.2 2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
             </material>
           </visual>
         </link>
       </model>
       
       <!-- Add a cube as an obstacle -->
       <model name="obstacle_cube">
         <pose>-2 0 0.5 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>1 1 1</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>1 1 1</size>
               </box>
             </geometry>
             <material>
               <ambient>0.2 0.8 0.2 1</ambient>
               <diffuse>0.2 0.8 0.2 1</diffuse>
             </material>
           </visual>
         </link>
       </model>
       
       <!-- Add a sphere as another object -->
       <model name="obstacle_sphere">
         <pose>2 1 0.5 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <sphere>
                 <radius>0.5</radius>
               </sphere>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <sphere>
                 <radius>0.5</radius>
               </sphere>
             </geometry>
             <material>
               <ambient>0.8 0.2 0.2 1</ambient>
               <diffuse>0.8 0.2 0.2 1</diffuse>
             </material>
           </visual>
         </link>
       </model>
     </world>
   </sdf>
   ```

7. Launch Gazebo with your custom world:
   ```bash
   gazebo --verbose ~/ros2_ws/gazebo_worlds/my_room.world
   ```

8. Explore the Gazebo interface:
   - Use the mouse to navigate: right-click and drag to move the camera
   - Left-click on objects to select them
   - Use the GUI tools to add more models from the database

#### Expected Outcome
Gazebo successfully launched with the custom world containing walls, cube, and sphere obstacles.

  </TabItem>
  <TabItem value="lab2">

### Lab 2: Robot Spawning and Control

#### Objective
Create a simple robot model and spawn it in Gazebo, then control it using ROS 2.

#### Required Components
- ROS 2 environment
- Gazebo
- Robot Model package

#### Steps
1. Create a robot description package for a simple differential drive robot:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python robot_gazebo_pkg --dependencies rclpy std_msgs geometry_msgs sensor_msgs
   ```

2. Create the URDF for a simple robot:
   ```bash
   mkdir -p ~/ros2_ws/src/robot_gazebo_pkg/urdf
   nano ~/ros2_ws/src/robot_gazebo_pkg/urdf/simple_robot.urdf
   ```

3. Add the following URDF definition:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
     <!-- Base Link -->
     <link name="base_link">
       <visual>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="5.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Left Wheel -->
     <joint name="left_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="left_wheel"/>
       <origin xyz="0 0.2 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>
     <link name="left_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <!-- Right Wheel -->
     <joint name="right_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="right_wheel"/>
       <origin xyz="0 -0.2 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>
     <link name="right_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <!-- Create a fixed joint to attach a camera to the front -->
     <joint name="camera_joint" type="fixed">
       <parent link="base_link"/>
       <child link="camera_link"/>
       <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
     </joint>
     <link name="camera_link">
       <visual>
         <geometry>
           <box size="0.05 0.05 0.05"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 0.8"/>
         </material>
       </visual>
     </link>
   </robot>
   ```

4. Create a launch file to spawn the robot in Gazebo:
   ```bash
   mkdir -p ~/ros2_ws/src/robot_gazebo_pkg/launch
   nano ~/ros2_ws/src/robot_gazebo_pkg/launch/spawn_robot.launch.py
   ```

5. Add the following launch file:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import ExecuteProcess, TimerAction
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Get the package share directory
       pkg_share = get_package_share_directory('robot_gazebo_pkg')
       
       # Path to the URDF file
       urdf_file = os.path.join(pkg_share, 'urdf', 'simple_robot.urdf')
       
       # Launch Gazebo with the custom world
       start_gazebo_cmd = ExecuteProcess(
           cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
           output='screen'
       )
       
       # Launch the robot state publisher
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           output='screen',
           parameters=[{
               'robot_description': open(urdf_file).read(),
               'publish_frequency': 50.0
           }]
       )
       
       # Launch the joint state publisher
       joint_state_publisher = Node(
           package='joint_state_publisher',
           executable='joint_state_publisher',
           parameters=[{
               'use_gui': False,
               'rate': 50
           }]
       )
       
       # Spawn the robot in Gazebo after a delay to ensure Gazebo is ready
       spawn_robot_cmd = TimerAction(
           period=5.0,
           actions=[
               ExecuteProcess(
                   cmd=[
                       'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                       '-entity', 'simple_robot',
                       '-file', urdf_file,
                       '-x', '0', '-y', '0', '-z', '0.1',  # Spawn 0.1m above ground
                       '-robot_namespace', 'simple_robot'
                   ],
                   output='screen'
               )
           ]
       )
       
       # Launch the controller (to control the wheels)
       diff_drive_spawner = TimerAction(
           period=7.0,
           actions=[
               Node(
                   package='controller_manager',
                   executable='spawner',
                   arguments=['diff_drive_controller', '-c', '/simple_robot/controller_manager'],
               )
           ]
       )
       
       # Load controllers
       controller_manager = TimerAction(
           period=6.0,
           actions=[
               Node(
                   package='controller_manager',
                   executable='ros2_control_node',
                   parameters=[os.path.join(pkg_share, 'config', 'robot_controllers.yaml')]
               )
           ]
       )
   
       ld = LaunchDescription()
       
       ld.add_action(start_gazebo_cmd)
       ld.add_action(robot_state_publisher)
       ld.add_action(joint_state_publisher)
       ld.add_action(spawn_robot_cmd)
       ld.add_action(diff_drive_spawner)
       ld.add_action(controller_manager)
       
       return ld
   ```

6. Since we referenced a controller configuration file, create it:
   ```bash
   mkdir -p ~/ros2_ws/src/robot_gazebo_pkg/config
   nano ~/ros2_ws/src/robot_gazebo_pkg/config/robot_controllers.yaml
   ```

7. Add the controller configuration:
   ```yaml
   controller_manager:
     ros__parameters:
       update_rate: 100  # Hz
   
       diff_drive_controller:
         type: diff_drive_controller/DiffDriveController
   
   diff_drive_controller:
     ros__parameters:
       left_wheel_names: ["left_wheel_joint"]
       right_wheel_names: ["right_wheel_joint"]
       
       wheel_separation: 0.4
       wheel_radius: 0.1
       
       # Publish rates
       publish_rate: 50.0
       odom_publish_rate: 50.0
       
       # Topic names
       cmd_vel_topic: "cmd_vel"
       odom_topic: "odom"
       pose_topic: "pose"
       twist_topic: "twist"
       
       # frame_ids
       odom_frame_id: "odom"
       base_frame_id: "base_link"
   ```

8. Actually, for a simpler approach, let's modify the launch file to work with the basic Gazebo interface:
   ```bash
   nano ~/ros2_ws/src/robot_gazebo_pkg/launch/spawn_robot.launch.py
   ```

9. Replace with a simplified version:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import ExecuteProcess, TimerAction
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Get the package share directory
       pkg_share = get_package_share_directory('robot_gazebo_pkg')
       
       # Path to the URDF file
       urdf_file = os.path.join(pkg_share, 'urdf', 'simple_robot.urdf')
       
       # Launch Gazebo
       start_gazebo_cmd = ExecuteProcess(
           cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
           output='screen'
       )
       
       # Launch the robot state publisher
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           output='screen',
           parameters=[{
               'robot_description': open(urdf_file).read(),
               'publish_frequency': 50.0
           }]
       )
       
       # Spawn the robot in Gazebo after a delay
       spawn_robot_cmd = TimerAction(
           period=5.0,
           actions=[
               ExecuteProcess(
                   cmd=[
                       'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                       '-entity', 'simple_robot',
                       '-file', urdf_file,
                       '-x', '0', '-y', '0', '-z', '0.1',
                       '-robot_namespace', 'simple_robot'
                   ],
                   output='screen'
               )
           ]
       )
       
       ld = LaunchDescription()
       
       ld.add_action(start_gazebo_cmd)
       ld.add_action(robot_state_publisher)
       ld.add_action(spawn_robot_cmd)
       
       return ld
   ```

10. Update setup.py to include launch files:
    ```bash
    nano ~/ros2_ws/src/robot_gazebo_pkg/setup.py
    ```

11. Add launch files to data_files:
    ```python
    import os
    from glob import glob
    from setuptools import setup

    package_name = 'robot_gazebo_pkg'

    setup(
        name=package_name,
        version='0.0.0',
        packages=[package_name],
        data_files=[
            ('share/ament_index/resource_index/packages',
                ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            # Include launch files
            (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
            # Include URDF files
            (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
            # Include config files
            (os.path.join('share', package_name, 'config'), glob('config/*')),
        ],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='Your Name',
        maintainer_email='you@example.com',
        description='Package for Gazebo robot simulation',
        license='Apache-2.0',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
            ],
        },
    )
    ```

12. Build the package:
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select robot_gazebo_pkg
    source install/setup.bash
    ```

13. Launch the simulation:
    ```bash
    ros2 launch robot_gazebo_pkg spawn_robot.launch.py
    ```

14. To control the robot, in another terminal, send velocity commands:
    ```bash
    # Move forward
    ros2 topic pub /simple_robot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.0}}"
    
    # Turn
    ros2 topic pub /simple_robot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.5}}"
    ```

#### Expected Outcome
Robot model successfully spawned in Gazebo and controllable via ROS 2 topics.

  </TabItem>
  <TabItem value="lab3">

### Lab 3: Sensor Simulation Integration

#### Objective
Add simulated sensors to the robot and interface them with ROS 2.

#### Required Components
- ROS 2 environment
- Gazebo
- The robot_gazebo_pkg from Lab 2

#### Steps
1. First, let's enhance our robot URDF to include a camera sensor:
   ```bash
   nano ~/ros2_ws/src/robot_gazebo_pkg/urdf/robot_with_camera.urdf
   ```

2. Add the following URDF with a camera sensor:
   ```xml
   <?xml version="1.0"?>
   <robot name="robot_with_camera" xmlns:xacro="http://www.ros.org/wiki/xacro">
     <!-- Base Link -->
     <link name="base_link">
       <visual>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="5.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Left Wheel -->
     <joint name="left_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="left_wheel"/>
       <origin xyz="0 0.2 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>
     <link name="left_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <!-- Right Wheel -->
     <joint name="right_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="right_wheel"/>
       <origin xyz="0 -0.2 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>
     <link name="right_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <!-- Camera Link -->
     <joint name="camera_joint" type="fixed">
       <parent link="base_link"/>
       <child link="camera_link"/>
       <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
     </joint>
     <link name="camera_link">
     </link>

     <!-- Camera Sensor -->
     <gazebo reference="camera_link">
       <sensor name="camera" type="camera">
         <always_on>true</always_on>
         <visualize>true</visualize>
         <update_rate>30</update_rate>
         <camera name="head">
           <horizontal_fov>1.089</horizontal_fov>
           <image>
             <width>640</width>
             <height>480</height>
             <format>R8G8B8</format>
           </image>
           <clip>
             <near>0.1</near>
             <far>100</far>
           </clip>
         </camera>
         <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
           <frame_name>camera_link</frame_name>
           <min_depth>0.1</min_depth>
           <max_depth>100.0</max_depth>
         </plugin>
       </sensor>
     </gazebo>
   </robot>
   ```

3. Also add a simple 2D LiDAR sensor to the robot:
   ```bash
   nano ~/ros2_ws/src/robot_gazebo_pkg/urdf/robot_with_sensors.urdf
   ```

4. Add the following URDF with both camera and LiDAR sensors:
   ```xml
   <?xml version="1.0"?>
   <robot name="robot_with_sensors" xmlns:xacro="http://www.ros.org/wiki/xacro">
     <!-- Base Link -->
     <link name="base_link">
       <visual>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.2" length="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="5.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Left Wheel -->
     <joint name="left_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="left_wheel"/>
       <origin xyz="0 0.2 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>
     <link name="left_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <!-- Right Wheel -->
     <joint name="right_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="right_wheel"/>
       <origin xyz="0 -0.2 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
     </joint>
     <link name="right_wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <!-- Camera Link -->
     <joint name="camera_joint" type="fixed">
       <parent link="base_link"/>
       <child link="camera_link"/>
       <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
     </joint>
     <link name="camera_link">
     </link>

     <!-- LiDAR Link -->
     <joint name="lidar_joint" type="fixed">
       <parent link="base_link"/>
       <child link="lidar_link"/>
       <origin xyz="0 0 0.15" rpy="0 0 0"/>
     </joint>
     <link name="lidar_link">
     </link>

     <!-- Camera Sensor -->
     <gazebo reference="camera_link">
       <sensor name="camera" type="camera">
         <always_on>true</always_on>
         <visualize>true</visualize>
         <update_rate>30</update_rate>
         <camera name="head">
           <horizontal_fov>1.089</horizontal_fov>
           <image>
             <width>640</width>
             <height>480</height>
             <format>R8G8B8</format>
           </image>
           <clip>
             <near>0.1</near>
             <far>100</far>
           </clip>
         </camera>
         <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
           <frame_name>camera_link</frame_name>
           <min_depth>0.1</min_depth>
           <max_depth>100.0</max_depth>
           <topic_name>/simple_robot/camera/image_raw</topic_name>
         </plugin>
       </sensor>
     </gazebo>

     <!-- LiDAR Sensor -->
     <gazebo reference="lidar_link">
       <sensor name="lidar" type="ray">
         <always_on>true</always_on>
         <visualize>true</visualize>
         <update_rate>10</update_rate>
         <ray>
           <scan>
             <horizontal>
               <samples>360</samples>
               <resolution>1.0</resolution>
               <min_angle>-3.14159</min_angle>
               <max_angle>3.14159</max_angle>
             </horizontal>
           </scan>
           <range>
             <min>0.1</min>
             <max>10.0</max>
             <resolution>0.01</resolution>
           </range>
         </ray>
         <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
           <ros>
             <namespace>/simple_robot</namespace>
             <remapping>~/out:=scan</remapping>
           </ros>
           <output_type>sensor_msgs/LaserScan</output_type>
         </plugin>
       </sensor>
     </gazebo>
   </robot>
   ```

5. Create a launch file for the robot with sensors:
   ```bash
   nano ~/ros2_ws/src/robot_gazebo_pkg/launch/spawn_robot_with_sensors.launch.py
   ```

6. Add the following launch file:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import ExecuteProcess, TimerAction
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Get the package share directory
       pkg_share = get_package_share_directory('robot_gazebo_pkg')
       
       # Path to the URDF file with sensors
       urdf_file = os.path.join(pkg_share, 'urdf', 'robot_with_sensors.urdf')
       
       # Launch Gazebo
       start_gazebo_cmd = ExecuteProcess(
           cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
           output='screen'
       )
       
       # Launch the robot state publisher
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           output='screen',
           parameters=[{
               'robot_description': open(urdf_file).read(),
               'publish_frequency': 50.0
           }]
       )
       
       # Spawn the robot in Gazebo after a delay
       spawn_robot_cmd = TimerAction(
           period=5.0,
           actions=[
               ExecuteProcess(
                   cmd=[
                       'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                       '-entity', 'simple_robot',
                       '-file', urdf_file,
                       '-x', '0', '-y', '0', '-z', '0.1',
                       '-robot_namespace', 'simple_robot'
                   ],
                   output='screen'
               )
           ]
       )
       
       # Create a simple node to move the robot around to test sensors
       sensor_test_node = TimerAction(
           period=10.0,
           actions=[
               Node(
                   package='robot_gazebo_pkg',
                   executable='sensor_test_node',
                   name='sensor_test_node'
               )
           ]
       )
       
       ld = LaunchDescription()
       
       ld.add_action(start_gazebo_cmd)
       ld.add_action(robot_state_publisher)
       ld.add_action(spawn_robot_cmd)
       # ld.add_action(sensor_test_node)  # Uncomment if we create this node
       
       return ld
   ```

7. Create a simple sensor viewer node to visualize the data:
   ```bash
   nano ~/ros2_ws/src/robot_gazebo_pkg/robot_gazebo_pkg/sensor_viewer_node.py
   ```

8. Add the following sensor viewer code:
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, LaserScan
   from cv_bridge import CvBridge
   import cv2

   class SensorViewerNode(Node):
       def __init__(self):
           super().__init__('sensor_viewer_node')
           
           # Initialize CvBridge
           self.bridge = CvBridge()
           
           # Create subscribers for sensor data
           self.image_subscription = self.create_subscription(
               Image,
               '/simple_robot/camera/image_raw',
               self.image_callback,
               10)
           
           self.scan_subscription = self.create_subscription(
               LaserScan,
               '/simple_robot/scan',
               self.scan_callback,
               10)
           
           self.get_logger().info('Sensor viewer node initialized')

       def image_callback(self, msg):
           # Convert the ROS Image message to OpenCV format
           try:
               cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
               # Display the image
               cv2.imshow("Camera View", cv_image)
               cv2.waitKey(1)
           except Exception as e:
               self.get_logger().error(f'Error converting image: {e}')

       def scan_callback(self, msg):
           # Log some basic information about the laser scan
           range_count = len(msg.ranges)
           min_range = min([r for r in msg.ranges if r != float('inf') and r > msg.range_min], default=msg.range_max)
           max_range = max([r for r in msg.ranges if r != float('inf') and r < msg.range_max], default=msg.range_min)
           
           self.get_logger().info(
               f'Laser scan: {range_count} points, '
               f'min range: {min_range:.2f}m, '
               f'max range: {max_range:.2f}m'
           )

   def main(args=None):
       rclpy.init(args=args)
       sensor_viewer_node = SensorViewerNode()
       
       try:
           rclpy.spin(sensor_viewer_node)
       except KeyboardInterrupt:
           pass
       
       sensor_viewer_node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

9. Update setup.py to include this new node:
    ```bash
    nano ~/ros2_ws/src/robot_gazebo_pkg/setup.py
    ```

10. Update the entry_points in setup.py:
    ```python
    entry_points={
        'console_scripts': [
            'sensor_viewer_node = robot_gazebo_pkg.sensor_viewer_node:main',
        ],
    }
    ```

11. Install cv_bridge (if not already installed):
    ```bash
    sudo apt update
    sudo apt install python3-opencv ros-humble-vision-opencv
    pip3 install opencv-python
    ```

12. Make the file executable and rebuild:
    ```bash
    chmod +x ~/ros2_ws/src/robot_gazebo_pkg/robot_gazebo_pkg/sensor_viewer_node.py
    cd ~/ros2_ws
    colcon build --packages-select robot_gazebo_pkg
    source install/setup.bash
    ```

13. Launch the simulation with sensors:
    ```bash
    ros2 launch robot_gazebo_pkg spawn_robot_with_sensors.launch.py
    ```

14. In another terminal, run the sensor viewer to see the data:
    ```bash
    ros2 run robot_gazebo_pkg sensor_viewer_node
    ```

15. Control the robot to see how the sensors respond:
    ```bash
    # Move forward to see objects in the camera and LiDAR
    ros2 topic pub /simple_robot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.0}}" -1
    ```

#### Expected Outcome
Robot with simulated sensors successfully running in Gazebo, with sensor data published to ROS 2 topics that can be viewed and processed.

  </TabItem>
</Tabs>

## Sim-to-Real Notes

- **Hardware considerations**: When designing simulations, carefully match sensor noise models and physical properties to real hardware specifications
- **Differences from simulation**: Real sensors often have different noise characteristics, latency, and accuracy compared to simulated ones
- **Practical tips**: Use domain randomization techniques to train robust algorithms that can handle both simulation and real-world variations

## Multiple Choice Questions

1. What is the primary purpose of Gazebo in robotics development?
   - A) To build robot hardware
   - B) To provide a physics simulation environment for testing robots
   - C) To program robot controllers
   - D) To manage ROS 2 packages

   **Correct Answer: B** - Gazebo provides a physics simulation environment for testing robots.

2. Which command is used to launch Gazebo with a specific world file?
   - A) gazebo start world_file.world
   - B) gazebo world_file.world
   - C) ros2 run gazebo world_file.world
   - D) launch gazebo world_file.world

   **Correct Answer: B** - gazebo world_file.world launches Gazebo with a specific world file.

3. What does URDF stand for in ROS/Gazebo context?
   - A) Universal Robot Description Format
   - B) Uniform Robot Definition File
   - C) Universal Robot Development Framework
   - D) Unified Robot Design Format

   **Correct Answer: A** - URDF stands for Universal Robot Description Format.

4. What is the purpose of the `<gazebo>` tag in URDF files?
   - A) To define robot joints
   - B) To specify simulation-specific properties for Gazebo
   - C) To create robot links
   - D) To set robot dimensions

   **Correct Answer: B** - The `<gazebo>` tag specifies simulation-specific properties for Gazebo.

5. Which ROS 2 package is commonly used to spawn entities in Gazebo?
   - A) gazebo_ros
   - B) robot_state_publisher
   - C) joint_state_publisher
   - D) tf2_ros

   **Correct Answer: A** - gazebo_ros contains tools to spawn entities in Gazebo.

6. How do you typically control a simulated robot in Gazebo using ROS 2?
   - A) Direct API calls to Gazebo
   - B) Through ROS 2 topics (e.g., /cmd_vel) that interface with Gazebo plugins
   - C) By modifying the URDF file
   - D) Using Gazebo's built-in controller only

   **Correct Answer: B** - Robot control happens through ROS 2 topics that interface with Gazebo plugins.

7. What is the SDF format used for in Gazebo?
   - A) Sensor Data Format
   - B) Simulation Description Format
   - C) System Definition File
   - D) Simple Data Format

   **Correct Answer: B** - SDF stands for Simulation Description Format, used for Gazebo worlds/models.

8. Which plugin would you use in Gazebo to simulate a camera sensor?
   - A) libgazebo_ros_laser.so
   - B) libgazebo_ros_imu.so
   - C) libgazebo_ros_camera.so
   - D) libgazebo_ros_diff_drive.so

   **Correct Answer: C** - libgazebo_ros_camera.so is used for camera simulation.

9. What is the purpose of the robot_state_publisher in Gazebo integration?
   - A) To publish sensor readings
   - B) To publish joint states and robot transforms
   - C) To control robot movement
   - D) To save robot configurations

   **Correct Answer: B** - robot_state_publisher publishes joint states and robot transforms.

10. In Gazebo, what is the purpose of the update_rate parameter in sensor definitions?
    - A) How fast the physics updates
    - B) How frequently the sensor publishes data
    - C) How often the graphics update
    - D) How fast the robot moves

    **Correct Answer: B** - update_rate specifies how frequently the sensor publishes data.

11. Which parameter determines the minimum valid range for a LiDAR sensor in Gazebo?
    - A) max_range
    - B) min_range
    - C) resolution
    - D) accuracy

    **Correct Answer: B** - min_range determines the minimum valid range for a LiDAR sensor.

12. How do you typically visualize the laser scan data from a simulated LiDAR?
    - A) Using RViz with a LaserScan display
    - B) Using a camera display in RViz
    - C) Using the Gazebo GUI
    - D) Through text output only

    **Correct Answer: A** - Using RViz with a LaserScan display is the standard approach.

13. What is the purpose of the joint_state_publisher in a typical Gazebo setup?
    - A) To publish transforms between robot links
    - B) To publish joint positions and velocities
    - C) To control robot movement
    - D) To publish sensor data

    **Correct Answer: B** - joint_state_publisher publishes joint positions and velocities.

14. Which command can be used to see the available topics from a running Gazebo simulation?
    - A) gazebo topics
    - B) ros2 node list
    - C) ros2 topic list
    - D) Both B and C

    **Correct Answer: D** - Both ros2 node list and ros2 topic list can be used to see available elements.

15. What is a common approach to test robot algorithms before deployment?
    - A) Direct hardware testing only
    - B) Simulation in Gazebo followed by hardware testing
    - C) Code simulation only
    - D) Visual inspection only

    **Correct Answer: B** - Simulation in Gazebo followed by hardware testing is the recommended approach.

## Further Reading

1. [Gazebo Simulation Guide](https://classic.gazebosim.org/tutorials) - Official Gazebo tutorials
2. [ROS 2 with Gazebo](https://github.com/ros-simulation/gazebo_ros_pkgs) - Gazebo ROS packages documentation
3. [URDF Tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF.html) - ROS URDF documentation
4. [Simulation Best Practices](https://docs.ros.org/en/humble/Tutorials/Advanced/Simulations.html) - ROS simulation guidelines
5. [Robot State Publisher](https://docs.ros.org/en/humble/p/robot_state_publisher/) - Documentation for robot state publisher
6. [Gazebo ROS2 Control](https://github.com/ros-simulation/gazebo_ros2_control) - Interface for controlling simulated robots

## Chapter Navigation

<div class="pagination-nav">
  <div class="pagination-nav__item pagination-nav__item--prev">
    <Link className="pagination-nav__link" to="/modules/module-1-ros/chapter-7-debugging-tools/">
      <div className="pagination-nav__sublabel">Previous</div>
      <div className="pagination-nav__label">← Chapter 7: Debugging and Visualization Tools</div>
    </Link>
  </div>
  <div class="pagination-nav__item pagination-nav__item--next">
    <Link className="pagination-nav__link" to="/modules/module-2-simulation/chapter-9-robot-modeling/">
      <div className="pagination-nav__sublabel">Next</div>
      <div className="pagination-nav__label">Chapter 9: Robot Modeling and URDF →</div>
    </Link>
  </div>
</div>