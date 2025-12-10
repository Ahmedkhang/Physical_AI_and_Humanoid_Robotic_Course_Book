---
id: chapter-6-packages-build
title: Chapter 6 - ROS 2 Packages and Build System
sidebar_label: Chapter 6
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Learning Outcomes

After completing this chapter, you will be able to:
1. Create and structure ROS 2 packages using both ament_cmake and ament_python
2. Understand the purpose and content of package.xml and CMakeLists.txt files
3. Implement proper dependency management in ROS 2 packages
4. Use colcon build system effectively for package compilation
5. Create custom message, service, and action definitions
6. Implement package testing and documentation standards
7. Follow ROS 2 package naming and organization conventions
8. Debug common build and package-related issues

## Gherkin Specifications

### Scenario 1: Package Creation
- **Given** a need to implement a specific robot functionality
- **When** creating a new ROS 2 package
- **Then** the package follows proper structure and conventions

### Scenario 2: Dependency Management
- **Given** a package with external dependencies
- **When** dependencies are properly declared in package.xml
- **Then** the package builds and runs correctly

### Scenario 3: Build System Execution
- **Given** a workspace with ROS 2 packages
- **When** colcon build is executed
- **Then** all packages compile successfully with proper linking

### Scenario 4: Message Definition
- **Given** a need to share custom data between nodes
- **When** a new message type is defined in .msg format
- **Then** the message can be used in publishers and subscribers

### Scenario 5: Package Distribution
- **Given** a completed ROS 2 package
- **When** the package is properly documented and tested
- **Then** it can be shared and reused by other developers

## Theory & Intuition

Think of ROS 2 packages like modules in a modular furniture system. Just as modular furniture pieces have standard connection points and follow specific dimensions to work together, ROS 2 packages have standard structures and interfaces that allow them to work together in a robot system.

Each package is like a self-contained piece of furniture - it has its own purpose (a bookshelf, desk, or chair) and comes with all the necessary components and instructions to function properly. The build system (colcon) is like the assembly instructions that tell you how to put the furniture together, ensuring all parts fit correctly and the final product functions as intended.

The package.xml file is like the product label that tells you what's inside the box, what other components it's compatible with, and who made it. The CMakeLists.txt (for C++ packages) or setup.py (for Python packages) is like the detailed assembly guide that specifies exactly how to combine the components.

## Core Concepts

<Tabs
  defaultValue="diagram"
  values={[
    {label: 'Package Structure', value: 'diagram'},
    {label: 'Build System Components', value: 'table'},
  ]}>
  <TabItem value="diagram">

```mermaid
graph TB
    subgraph "ROS 2 Package"
        A[package.xml]
        B[CMakeLists.txt/setup.py]
        C[src/ or ros2_pkg/]
        D[include/ (for C++)]
        E[config/, launch/, etc.]
    end
    
    subgraph "Build Process"
        F[colcon build]
        G[ament_cmake/ament_python]
        H[Generated Files]
    end
    
    A --> F
    B --> F
    C --> F
    D --> F
    E --> F
    F --> G
    G --> H
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e8f5e8
    style F fill:#f3e5f5
```

  </TabItem>
  <TabItem value="table">

| File/Directory | Purpose | Required |
|----------------|---------|----------|
| package.xml | Package metadata and dependencies | Yes |
| CMakeLists.txt | Build instructions for C++ packages | Yes (for C++) |
| setup.py | Build instructions for Python packages | Yes (for Python) |
| src/ | Source code files | No |
| include/ | Header files (C++) | No (for C++) |
| launch/ | Launch files | No |
| config/ | Configuration files | No |

  </TabItem>
</Tabs>

## Hands-On Labs

<Tabs
  defaultValue="lab1"
  values={[
    {label: 'Lab 1: Creating a Basic Package', value: 'lab1'},
    {label: 'Lab 2: Message and Service Definitions', value: 'lab2'},
    {label: 'Lab 3: Package Dependencies and Build Process', value: 'lab3'},
  ]}>
  <TabItem value="lab1">

### Lab 1: Creating a Basic Package

#### Objective
Create a basic ROS 2 package with proper structure and functionality.

#### Required Components
- ROS 2 environment
- Text editor
- Terminal access

#### Steps
1. Create a new package using the command line tool:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python basic_pkg --dependencies rclpy std_msgs geometry_msgs
   ```

2. Explore the created package structure:
   ```bash
   ls -la ~/ros2_ws/src/basic_pkg/
   ```

3. Modify the package.xml to add more details:
   ```bash
   nano ~/ros2_ws/src/basic_pkg/package.xml
   ```

4. Update the package.xml file:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>basic_pkg</name>
     <version>0.0.0</version>
     <description>Basic package for learning ROS 2 concepts</description>
     <maintainer email="you@example.com">Your Name</maintainer>
     <license>Apache-2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>geometry_msgs</depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

5. Create a simple node in the package:
   ```bash
   nano ~/ros2_ws/src/basic_pkg/basic_pkg/basic_node.py
   ```

6. Add the following code:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist

   class BasicNode(Node):
       def __init__(self):
           super().__init__('basic_node')
           
           # Publisher
           self.publisher = self.create_publisher(String, 'basic_topic', 10)
           
           # Subscriber
           self.subscription = self.create_subscription(
               Twist,
               'cmd_vel',
               self.cmd_vel_callback,
               10)
           
           # Timer
           timer_period = 1.0  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           
           self.get_logger().info('Basic node initialized')

       def timer_callback(self):
           msg = String()
           msg.data = f'Basic node running at {self.get_clock().now().nanoseconds}'
           self.publisher.publish(msg)
           self.get_logger().info(f'Published: {msg.data}')

       def cmd_vel_callback(self, msg):
           self.get_logger().info(f'Received cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

   def main(args=None):
       rclpy.init(args=args)
       basic_node = BasicNode()
       rclpy.spin(basic_node)
       basic_node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

7. Make the node executable:
   ```bash
   chmod +x ~/ros2_ws/src/basic_pkg/basic_pkg/basic_node.py
   ```

8. Update setup.py to register the executable:
   ```bash
   nano ~/ros2_ws/src/basic_pkg/setup.py
   ```

9. Update the setup.py file:
   ```python
   from setuptools import setup

   package_name = 'basic_pkg'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='you@example.com',
       description='Basic package for learning ROS 2 concepts',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'basic_node = basic_pkg.basic_node:main',
           ],
       },
   )
   ```

10. Build the package:
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select basic_pkg
    source install/setup.bash
    ```

11. Run the basic node:
    ```bash
    ros2 run basic_pkg basic_node
    ```

#### Expected Outcome
Package successfully created with proper structure and the node running correctly.

  </TabItem>
  <TabItem value="lab2">

### Lab 2: Message and Service Definitions

#### Objective
Create custom message and service definitions in a ROS 2 package.

#### Required Components
- ROS 2 environment
- Text editor
- Terminal access

#### Steps
1. Create a new package for custom interfaces:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_cmake interface_pkg
   ```

2. Create directories for different interface types:
   ```bash
   mkdir -p ~/ros2_ws/src/interface_pkg/msg
   mkdir -p ~/ros2_ws/src/interface_pkg/srv
   ```

3. Create a custom message definition:
   ```bash
   nano ~/ros2_ws/src/interface_pkg/msg/RobotStatus.msg
   ```

4. Add the message definition:
   ```text
   # Custom message for robot status
   string robot_name
   float64 battery_level
   bool is_charging
   geometry_msgs/Pose current_pose
   bool[] joint_states
   ```

5. Create a custom service definition:
   ```bash
   nano ~/ros2_ws/src/interface_pkg/srv/RobotControl.srv
   ```

6. Add the service definition:
   ```text
   # Request: command type and value
   string command
   float64 value
   ---
   # Response: success status and message
   bool success
   string message
   ```

7. Update CMakeLists.txt to include message and service generation:
   ```bash
   nano ~/ros2_ws/src/interface_pkg/CMakeLists.txt
   ```

8. Add the following content before the ament_package() line:
   ```cmake
   find_package(rosidl_default_generators REQUIRED)
   find_package(geometry_msgs REQUIRED)

   rosidl_generate_interfaces(${PROJECT_NAME}
     "msg/RobotStatus.msg"
     "srv/RobotControl.srv"
     DEPENDENCIES geometry_msgs
   )

   ament_export_dependencies(rosidl_default_runtime)
   ```

9. Update package.xml to include required dependencies:
   ```bash
   nano ~/ros2_ws/src/interface_pkg/package.xml
   ```

10. Add these dependencies inside the `<package>` tag:
    ```xml
    <depend>geometry_msgs</depend>
    <buildtool_depend>rosidl_default_generators</buildtool_depend>
    <exec_depend>rosidl_default_runtime</exec_depend>
    <member_of_group>rosidl_interface_packages</member_of_group>
    ```

11. Create a Python package for using the custom interfaces:
    ```bash
    ros2 pkg create --build-type ament_python interface_user_pkg --dependencies rclpy
    ```

12. Create a node that uses the custom interface:
    ```bash
    nano ~/ros2_ws/src/interface_user_pkg/interface_user_pkg/user_node.py
    ```

13. Add the following code:
    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    # Import custom interfaces
    from interface_pkg.msg import RobotStatus
    from interface_pkg.srv import RobotControl

    class InterfaceUserNode(Node):
        def __init__(self):
            super().__init__('interface_user_node')
            
            # Publisher for custom message
            self.status_publisher = self.create_publisher(RobotStatus, 'robot_status', 10)
            
            # Service server for custom service
            self.srv = self.create_service(RobotControl, 'robot_control', self.control_callback)
            
            # Timer to periodically publish robot status
            timer_period = 1.0  # seconds
            self.timer = self.create_timer(timer_period, self.publish_status)
            
            self.get_logger().info('Interface user node initialized')

        def publish_status(self):
            msg = RobotStatus()
            msg.robot_name = "TestRobot"
            msg.battery_level = 85.0
            msg.is_charging = False
            msg.joint_states = [True, True, False, True]
            
            # Set pose (just example values)
            msg.current_pose.position.x = 1.0
            msg.current_pose.position.y = 2.0
            msg.current_pose.orientation.w = 1.0
            
            self.status_publisher.publish(msg)
            self.get_logger().info(f'Published robot status for {msg.robot_name}')

        def control_callback(self, request, response):
            self.get_logger().info(f'Received command: {request.command} with value: {request.value}')
            
            # Process the command (simplified)
            if request.command in ["move", "rotate", "stop", "charge"]:
                response.success = True
                response.message = f'Command {request.command} executed successfully'
            else:
                response.success = False
                response.message = f'Unknown command: {request.command}'
                
            return response

    def main(args=None):
        rclpy.init(args=args)
        interface_user_node = InterfaceUserNode()
        rclpy.spin(interface_user_node)
        interface_user_node.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

14. Make the file executable and update setup.py:
    ```bash
    chmod +x ~/ros2_ws/src/interface_user_pkg/interface_user_pkg/user_node.py
    nano ~/ros2_ws/src/interface_user_pkg/setup.py
    ```

15. Update setup.py with the entry point:
    ```python
    from setuptools import setup

    package_name = 'interface_user_pkg'

    setup(
        name=package_name,
        version='0.0.0',
        packages=[package_name],
        data_files=[
            ('share/ament_index/resource_index/packages',
                ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
        ],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='Your Name',
        maintainer_email='you@example.com',
        description='Package to use custom interfaces',
        license='Apache-2.0',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'user_node = interface_user_pkg.user_node:main',
            ],
        },
    )
    ```

16. Build both packages:
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select interface_pkg interface_user_pkg
    source install/setup.bash
    ```

17. Run the interface user node:
    ```bash
    ros2 run interface_user_pkg user_node
    ```

18. In another terminal, test the custom service:
    ```bash
    ros2 service call /robot_control interface_pkg/srv/RobotControl "{
      command: 'move',
      value: 1.0
    }"
    ```

#### Expected Outcome
Custom message and service types successfully defined and used by nodes.

  </TabItem>
  <TabItem value="lab3">

### Lab 3: Package Dependencies and Build Process

#### Objective
Manage complex dependencies and use the colcon build system effectively.

#### Required Components
- ROS 2 environment
- Text editor
- Terminal access
- The packages created in previous labs

#### Steps
1. Create a new package that depends on the previous packages:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python dependency_pkg --dependencies rclpy std_msgs basic_pkg interface_pkg
   ```

2. Create a node that uses multiple dependencies:
   ```bash
   nano ~/ros2_ws/src/dependency_pkg/dependency_pkg/dependency_node.py
   ```

3. Add the following code:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   # Import from our custom packages
   from basic_pkg.basic_node import BasicNode
   from interface_pkg.msg import RobotStatus
   from interface_pkg.srv import RobotControl

   class DependencyNode(Node):
       def __init__(self):
           super().__init__('dependency_node')
           
           # Publisher
           self.publisher = self.create_publisher(String, 'dependency_topic', 10)
           
           # Subscription to the RobotStatus from interface_pkg
           self.status_sub = self.create_subscription(
               RobotStatus,
               'robot_status',
               self.status_callback,
               10)
           
           # Client for the RobotControl service
           self.cli = self.create_client(RobotControl, 'robot_control')
           
           # Timer
           timer_period = 2.0  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           
           self.get_logger().info('Dependency node initialized, waiting for service...')

           while not self.cli.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Service not available, waiting again...')

           self.get_logger().info('Connected to robot_control service')

       def status_callback(self, msg):
           self.get_logger().info(
               f'Dependency node received status from {msg.robot_name}, '
               f'battery: {msg.battery_level}%'
           )

       def timer_callback(self):
           msg = String()
           msg.data = 'Dependency node is managing multiple packages'
           self.publisher.publish(msg)
           
           # Call the service
           self.call_control_service()

       def call_control_service(self):
           request = RobotControl.Request()
           request.command = 'status'
           request.value = 0.0
           
           future = self.cli.call_async(request)
           # We won't wait for response in this example

   def main(args=None):
       rclpy.init(args=args)
       dependency_node = DependencyNode()
       rclpy.spin(dependency_node)
       dependency_node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. Make the file executable:
   ```bash
   chmod +x ~/ros2_ws/src/dependency_pkg/dependency_pkg/dependency_node.py
   nano ~/ros2_ws/src/dependency_pkg/setup.py
   ```

5. Update setup.py:
   ```python
   from setuptools import setup

   package_name = 'dependency_pkg'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='you@example.com',
       description='Package that depends on other packages',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'dependency_node = dependency_pkg.dependency_node:main',
           ],
       },
   )
   ```

6. Update package.xml to properly declare dependencies:
   ```bash
   nano ~/ros2_ws/src/dependency_pkg/package.xml
   ```

7. Complete the package.xml file:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>dependency_pkg</name>
     <version>0.0.0</version>
     <description>Package that depends on other packages</description>
     <maintainer email="you@example.com">Your Name</maintainer>
     <license>Apache-2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>basic_pkg</depend>
     <depend>interface_pkg</depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

8. Build the entire workspace (this time without specific package selection to test dependencies):
   ```bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

9. Test build only dependencies:
   ```bash
   colcon build --packages-up-to dependency_pkg
   ```

10. Use colcon to test for the package:
    ```bash
    colcon test --packages-select dependency_pkg
    ```

11. Check test results:
    ```bash
    colcon test-result --all
    ```

#### Expected Outcome
Package with dependencies successfully built using colcon, demonstrating proper dependency management.

  </TabItem>
</Tabs>

## Sim-to-Real Notes

- **Hardware considerations**: When building packages for deployment on robot hardware, use cross-compilation techniques or ensure the target platform has all necessary build dependencies
- **Differences from simulation**: Real robots may require additional hardware-specific packages that should be properly declared and managed
- **Practical tips**: Use package.xml to clearly document dependencies to ensure other developers can easily build and run your packages

## Multiple Choice Questions

1. What is the primary build tool for ROS 2 packages?
   - A) catkin_make
   - B) cmake
   - C) colcon
   - D) make

   **Correct Answer: C** - colcon is the primary build tool for ROS 2 packages.

2. Which file contains metadata and dependencies for a ROS 2 package?
   - A) CMakeLists.txt
   - B) package.xml
   - C) setup.py
   - D) manifest.xml

   **Correct Answer: B** - package.xml contains metadata and dependencies for a ROS 2 package.

3. What command is used to create a new ROS 2 package?
   - A) catkin_create_pkg
   - B) ros2 create package_name
   - C) ros2 pkg create package_name
   - D) new_ros_package

   **Correct Answer: C** - ros2 pkg create package_name is the correct command.

4. Which build type should you use for a Python-based ROS 2 package?
   - A) ament_cmake
   - B) ament_python
   - C) cmake_python
   - D) python_pkg

   **Correct Answer: B** - ament_python is used for Python-based packages.

5. How do you specify the build type in package.xml?
   - A) <build_type>ament_python</build_type>
   - B) <type>ament_python</type>
   - C) <build>ament_python</build>
   - D) <format>ament_python</format>

   **Correct Answer: A** - <build_type>ament_python</build_type> specifies the build type.

6. What is the purpose of the CMakeLists.txt file?
   - A) To store package metadata
   - B) To specify build instructions for C++ packages
   - C) To define Python dependencies
   - D) To list launch files

   **Correct Answer: B** - CMakeLists.txt specifies build instructions for C++ packages.

7. Which command builds only a specific package?
   - A) colcon build --package package_name
   - B) colcon build --select package_name
   - C) colcon build --packages-select package_name
   - D) build package_name

   **Correct Answer: C** - colcon build --packages-select package_name builds only the specific package.

8. How can you build a package and all packages that depend on it?
   - A) colcon build --packages-above package_name
   - B) colcon build --packages-below package_name
   - C) colcon build --packages-up-to package_name
   - D) colcon build --with-dependencies package_name

   **Correct Answer: B** - colcon build --packages-below package_name builds the package and all that depend on it.

9. What is the purpose of the `<exec_depend>` tag in package.xml?
   - A) To specify build-time dependencies
   - B) To specify runtime dependencies
   - C) To specify test dependencies
   - D) To specify optional dependencies

   **Correct Answer: B** - `<exec_depend>` specifies runtime dependencies.

10. Where should you place custom message definitions?
    - A) In the src/ directory
    - B) In the msg/ directory
    - C) In the include/ directory
    - D) In the config/ directory

    **Correct Answer: B** - Custom message definitions go in the msg/ directory.

11. How do you generate code from custom message definitions?
    - A) Using a custom script
    - B) Using rosidl_generate_interfaces in CMakeLists.txt
    - C) Manually writing the code
    - D) Using a ROS 1 tool

    **Correct Answer: B** - Using rosidl_generate_interfaces in CMakeLists.txt generates code from custom message definitions.

12. What is the purpose of setup.py in Python packages?
    - A) To define C++ build instructions
    - B) To specify Python package build instructions and entry points
    - C) To store configuration parameters
    - D) To define message types

    **Correct Answer: B** - setup.py specifies Python package build instructions and entry points.

13. Which command lists all available packages in the workspace?
    - A) ros2 pkg list
    - B) colcon list
    - C) ament list
    - D) ros2 list packages

    **Correct Answer: A** - ros2 pkg list shows all available packages.

14. How do you run tests for a specific package?
    - A) colcon test package_name
    - B) colcon test --packages-select package_name
    - C) ros2 test package_name
    - D) ament test package_name

    **Correct Answer: B** - colcon test --packages-select package_name runs tests for a specific package.

15. What is the recommended folder for launch files in a ROS 2 package?
    - A) scripts/
    - B) config/
    - C) launch/
    - D) resources/

    **Correct Answer: C** - Launch files should be placed in the launch/ directory.

## Further Reading

1. [ROS 2 Package Creation](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html) - Official guide to creating packages
2. [Colcon Build Tool](https://colcon.readthedocs.io/en/released/) - Documentation for the colcon build system
3. [Package XML Format](https://docs.ros.org/en/humble/How-To-Guides/Ament-Creation-Framework.html) - Guide to package.xml format
4. [Custom Message Definitions](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html) - Tutorial on creating custom messages
5. [ROS 2 Dependencies](https://docs.ros.org/en/humble/How-To-Guides/Node-Dependencies.html) - Managing dependencies in ROS 2
6. [ament_cmake vs ament_python](https://docs.ros.org/en/humble/How-To-Guides/Ament-CMake-Packages.html) - Choosing the right build system