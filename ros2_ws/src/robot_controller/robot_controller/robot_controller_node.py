#!/usr/bin/env python3

# Copyright 2025 Your Name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Import message types
from sensor_msgs.msg import Image         # For image subscription
from sensor_msgs.msg import JointState    # For joint state subscription
from std_msgs.msg import String           # For task description subscription
# Choose the appropriate message type for your joint commands.
# trajectory_msgs/JointTrajectory is common for controllers expecting sequences.
# sensor_msgs/JointState might be used for simpler position/velocity commands.
# Use the one that matches your robot's control interface.
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from sensor_msgs.msg import JointState as JointCommand # Alternative if needed

# --- Configuration Variables ---
# --- Subscription Topics ---
IMAGE_TOPIC = "/camera/image_raw"         # Change to your image topic
TASK_DESCRIPTION_TOPIC = "/task/description" # Change to your task description topic
JOINT_STATES_TOPIC = "/joint_states"       # Change to your joint_states topic

# --- Publication Topics ---
JOINT_COMMAND_TOPIC = "/joint_trajectory_controller/joint_trajectory" # Change to your command topic
# JOINT_COMMAND_TOPIC = "/joint_command" # Alternative if using JointState for commands

# --- Other Parameters ---
NODE_NAME = "robot_controller_node"
DEFAULT_QOS_DEPTH = 10
PROCESSING_TIMER_PERIOD = 0.1 # seconds (e.g., process data at 10 Hz)

class RobotControllerNode(Node):
    """
    A ROS 2 node that subscribes to image, task description, and joint states,
    processes the information, and publishes joint commands.
    """
    def __init__(self):
        """
        Initializes the node, subscribers, publisher, and internal state.
        """
        super().__init__(NODE_NAME)
        self.get_logger().info(f"Initializing {NODE_NAME}...")

        # --- QoS Profiles (customize if needed) ---
        # Best effort for sensors often makes sense if occasional drops are okay
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # Only keep the latest
        )
        # Reliable for commands and potentially state if needed for control loop
        reliable_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=DEFAULT_QOS_DEPTH
        )
        # Keep last state/task description
        state_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL # Get last published message on connection
        )


        # --- Internal State Variables ---
        self.latest_image: Image | None = None
        self.latest_task_description: String | None = None
        self.latest_joint_states: JointState | None = None
        self.joint_names = [] # Store joint names from the first joint_states message

        # --- Subscribers ---
        self.image_subscriber = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            sensor_qos_profile # Use appropriate QoS
        )
        self.get_logger().info(f"Subscribed to image topic: '{IMAGE_TOPIC}'")

        self.task_subscriber = self.create_subscription(
            String,
            TASK_DESCRIPTION_TOPIC,
            self.task_callback,
            state_qos_profile # Use appropriate QoS
        )
        self.get_logger().info(f"Subscribed to task description topic: '{TASK_DESCRIPTION_TOPIC}'")

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self.joint_state_callback,
            reliable_qos_profile # Use appropriate QoS
        )
        self.get_logger().info(f"Subscribed to joint states topic: '{JOINT_STATES_TOPIC}'")

        # --- Publisher ---
        self.joint_command_publisher = self.create_publisher(
            JointTrajectory, # Or JointCommand if using JointState
            JOINT_COMMAND_TOPIC,
            reliable_qos_profile # Use appropriate QoS
        )
        self.get_logger().info(f"Publishing joint commands to: '{JOINT_COMMAND_TOPIC}'")

        # --- Timer for processing data and publishing commands ---
        self.processing_timer = self.create_timer(
            PROCESSING_TIMER_PERIOD,
            self.process_and_publish_callback
        )
        self.get_logger().info(f"Processing timer set to {PROCESSING_TIMER_PERIOD} seconds.")

        self.get_logger().info(f"{NODE_NAME} initialization complete.")

    # --- Callback Functions ---
    def image_callback(self, msg: Image):
        """
        Callback function for the image subscriber.
        Stores the latest image message.
        """
        # self.get_logger().debug(f"Received image: {msg.header.stamp}")
        self.latest_image = msg
        # Add minimal processing here if needed immediately, but heavy processing
        # is better done in the timer callback.

    def task_callback(self, msg: String):
        """
        Callback function for the task description subscriber.
        Stores the latest task description message.
        """
        self.get_logger().info(f"Received task description: '{msg.data}'")
        self.latest_task_description = msg

    def joint_state_callback(self, msg: JointState):
        """
        Callback function for the joint state subscriber.
        Stores the latest joint state message and joint names if not already stored.
        """
        # self.get_logger().debug(f"Received joint states: {msg.header.stamp}")
        self.latest_joint_states = msg
        # Store joint names when first message is received
        if not self.joint_names and msg.name:
            self.joint_names = msg.name
            self.get_logger().info(f"Received joint names: {self.joint_names}")

    # --- Processing and Publishing Logic ---
    def process_and_publish_callback(self):
        """
        Timer callback where core logic resides.
        Processes received data and publishes joint commands.
        """
        # --- Check if all necessary data is available ---
        if (self.latest_image is None or
            self.latest_task_description is None or
            self.latest_joint_states is None or
                not self.joint_names):
            # self.get_logger().warn("Waiting for necessary data...", throttle_skip_first=True, throttle_duration_sec=5)
            # Optional: Add throttling to avoid flooding logs
            return

        # --- Core Logic Placeholder ---
        # Access the latest data:
        current_image = self.latest_image
        current_task = self.latest_task_description.data
        current_joint_states = self.latest_joint_states

        # TODO: Implement your robot control logic here.
        # This logic should:
        # 1. Analyze the `current_image` (e.g., detect objects, estimate poses).
        # 2. Interpret the `current_task` string.
        # 3. Consider the `current_joint_states` (positions, velocities).
        # 4. Calculate the desired `next_joint_positions`, `velocities`, etc.
        # Example: Simple logic to move the first joint slightly
        try:
            next_joint_positions = list(current_joint_states.position) # Start from current positions
            # Make sure we have joints and positions
            if next_joint_positions and self.joint_names:
                 # Modify the first joint position slightly (example)
                 # Replace this with your actual control calculation!
                first_joint_index = self.joint_names.index(self.joint_names[0]) # Get index safely
                next_joint_positions[first_joint_index] += 0.01
                # Ensure the command has the same number of positions as joint names
                if len(next_joint_positions) != len(self.joint_names):
                     self.get_logger().error("Mismatch between calculated positions and joint names count.")
                     return
            else:
                self.get_logger().warn("No joint positions or names available to command.")
                return

        except Exception as e:
            self.get_logger().error(f"Error during command calculation: {e}")
            return

        # --- Construct the Joint Command Message ---
        # Using JointTrajectory example
        joint_trajectory_msg = JointTrajectory()
        joint_trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        joint_trajectory_msg.joint_names = self.joint_names

        # Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = next_joint_positions
        # Optionally set velocities, accelerations, effort
        # point.velocities = [0.0] * len(self.joint_names) # Example: command zero velocity
        # point.accelerations = [0.0] * len(self.joint_names)
        # point.effort = [0.0] * len(self.joint_names)
        point.time_from_start = rclpy.duration.Duration(seconds=0.1).to_msg() # Time to reach point

        joint_trajectory_msg.points.append(point)

        # --- Publish the Command ---
        try:
            self.joint_command_publisher.publish(joint_trajectory_msg)
            # self.get_logger().debug(f"Published joint command: {next_joint_positions}")
        except Exception as e:
            self.get_logger().error(f"Failed to publish joint command: {e}")

        # Optional: Clear latest image if you only want to process new ones
        # self.latest_image = None


def main(args=None):
    """
    Main function to initialize ROS, create the node, spin, and shutdown.
    """
    rclpy.init(args=args)
    robot_controller_node = None
    try:
        robot_controller_node = RobotControllerNode()
        rclpy.spin(robot_controller_node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        if robot_controller_node:
            robot_controller_node.get_logger().error(f"Unhandled exception: {e}")
        else:
            print(f"Unhandled exception during node creation: {e}")
    finally:
        # Cleanup
        if robot_controller_node:
            robot_controller_node.destroy_node()
            print(f"Node {NODE_NAME} destroyed.")
        if rclpy.ok():
            rclpy.shutdown()
            print("rclpy shut down.")

if __name__ == '__main__':
    main()