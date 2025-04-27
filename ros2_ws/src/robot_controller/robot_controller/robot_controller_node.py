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
import numpy as np
import torch # Import torch
from typing import Dict, Any, List, Tuple

# Import ROS message types
from sensor_msgs.msg import Image         # For image subscription
from sensor_msgs.msg import JointState    # For joint state subscription
from std_msgs.msg import String           # For task description subscription
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Import CV Bridge for image conversion
from cv_bridge import CvBridge, CvBridgeError
import cv2 # OpenCV for potential resizing/processing

# Import sleep to keep the hz of the action publishing
from time import sleep

# Import the Gr00t inference client library
try:
    from gr00t.eval.service import ExternalRobotInferenceClient
except ImportError:
    print("Error: Could not import 'gr00t.eval.service'.")
    print("Please ensure the library is installed and accessible.")
    # Optionally exit or raise a more specific error
    import sys
    sys.exit(1)


# --- Configuration Variables ---
# --- ROS Topics ---
IMAGE_TOPIC = "/rgb"
TASK_DESCRIPTION_TOPIC = "/task"
STATE_TOPIC = "/joint_states"
ACTION_TOPIC = "/joint_command"

# --- Node and Timing ---
NODE_NAME = "gr00t_robot_controller_node"
DEFAULT_QOS_DEPTH = 10
PROCESSING_TIMER_PERIOD = 0.1 # seconds (adjust as needed for inference speed)

# --- Gr00t Inference Server ---
INFERENCE_SERVER_HOST = "localhost"
INFERENCE_SERVER_PORT = 5555
# Define expected image size for the policy
POLICY_IMAGE_WIDTH = 1280
POLICY_IMAGE_HEIGHT = 720

# --- Constants based on provided structure ---
ARM_STATE_INDICES = slice(0, 5)
GRIPPER_STATE_INDEX = 5 # Index for the single gripper value
ARM_ACTION_INDICES = slice(0, 5)
GRIPPER_ACTION_INDEX = 5

# Helper function to check joint count
EXPECTED_JOINT_COUNT = 6 # 5 arm + 1 gripper

#==============================================================================
# GR00T Inference Client Wrapper Class
#==============================================================================
class Gr00tRobotInferenceClient:
    """
    Wraps the ExternalRobotInferenceClient to handle data formatting
    specifically for the ROS node context.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        img_size: Tuple[int, int] = (POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH) # H, W
    ):
        """
        Initializes the connection to the external inference server.

        Args:
            host: Hostname or IP address of the inference server.
            port: Port number of the inference server.
            img_size: Expected image size (height, width) for the policy model.
        """
        self.img_size = img_size
        print(f"Attempting to connect to inference server at {host}:{port}...")
        try:
            self.policy = ExternalRobotInferenceClient(host=host, port=port)
            print("Successfully connected to inference server.")
        except Exception as e:
            print(f"Failed to connect to inference server: {e}")
            raise  

        # Determine device (prefer GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


    def format_observation(self,
                           img_np: np.ndarray,
                           joint_state_np: np.ndarray,
                           task_description: str
                           ) -> Dict[str, Any]:
        """
        Formats the raw numpy data and task description into the dictionary
        expected by the ExternalRobotInferenceClient.

        Args:
            img_np: Image as a NumPy array (H, W, C). Assumed BGR from cv_bridge.
            joint_state_np: Joint positions as a 1D NumPy array.
            task_description: The language instruction string.

        Returns:
            A dictionary formatted for the policy's get_action method.

        Raises:
            ValueError: If joint_state_np doesn't have the expected length.
            TypeError: If input types are incorrect (should be handled by type hints).
        """
        if joint_state_np.shape[0] < EXPECTED_JOINT_COUNT:
            raise ValueError(f"Joint state has {joint_state_np.shape[0]} elements,"
                             f" expected at least {EXPECTED_JOINT_COUNT}.")

        # Construct Observation Dictionary with Torch Tensors
        obs_dict = {
            "video.webcam": img_np[np.newaxis, :, :, :],
            "state.single_arm": joint_state_np[ARM_STATE_INDICES][np.newaxis, :].astype(np.float64),
            "state.gripper": joint_state_np[GRIPPER_STATE_INDEX:GRIPPER_STATE_INDEX+1][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [task_description],
        }
        return obs_dict

    def get_action(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the formatted observation dictionary to the inference server
        and returns the raw action chunk.

        Args:
            obs_dict: The dictionary prepared by format_observation.

        Returns:
            The raw action dictionary received from the server.
        """
        # Add timing if needed: start_time = time.time()
        raw_action_chunk = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return raw_action_chunk

    def extract_joint_commands(self, raw_action_chunk: Dict[str, Any]) -> np.ndarray | None:
        """
        Extracts and combines arm and gripper actions from the server response.

        Args:
            raw_action_chunk: The dictionary returned by self.policy.get_action.

        Returns:
            A NumPy array containing the combined joint commands (arm + gripper),
            or None if extraction fails.
        """
        try:
            if "action.single_arm" in raw_action_chunk and "action.gripper" in raw_action_chunk:
                arm_action = raw_action_chunk["action.single_arm"]
                gripper_action = raw_action_chunk["action.gripper"]

                combined_action = np.append(arm_action, gripper_action[:, np.newaxis], axis=1)

                if combined_action[0].shape[0] != EXPECTED_JOINT_COUNT:
                     print(f"Warning: Extracted action has {combined_action[0].shape[0]} elements," \
                           f" expected {EXPECTED_JOINT_COUNT}.")
                     # Decide how to handle this - return None, raise error, or try to use?
                     return None

                return combined_action.astype(np.float64) # ROS often uses float64
            else:
                print("Error: 'action.single_arm' or 'action.gripper' not found in server response.")
                return None
        except Exception as e:
            print(f"Error extracting actions from server response: {e}")
            print(f"Received action chunk: {raw_action_chunk}")
            return None

#==============================================================================
# ROS 2 Node Class
#==============================================================================
class RobotControllerNode(Node):
    """
    ROS 2 node using Gr00t client for inference-based control.
    """
    def __init__(self):
        super().__init__(NODE_NAME)
        self.get_logger().info(f"Initializing {NODE_NAME}...")

        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- Gr00t Client ---
        try:
            self.gr00t_client = Gr00tRobotInferenceClient(
                host=INFERENCE_SERVER_HOST,
                port=INFERENCE_SERVER_PORT,
                img_size=(POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH)
            )
        except Exception as e:
             self.get_logger().fatal(f"Could not initialize Gr00t client: {e}. Shutting down.")
             # A more robust approach might involve retrying or entering a safe mode
             raise # Reraise to stop node creation if connection fails

        # --- QoS Profiles ---
        sensor_qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        reliable_qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=DEFAULT_QOS_DEPTH)
        state_qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # --- Internal State Variables ---
        self.latest_image_msg: Image | None = None
        self.latest_task_description_msg: String | None = None
        self.latest_joint_states_msg: JointState | None = None
        self.joint_names: List[str] = [] # Store joint names from the first joint_states message
        self.last_command_time = self.get_clock().now()

        # --- Subscribers ---
        self.image_subscriber = self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, sensor_qos_profile)
        self.task_subscriber = self.create_subscription(String, TASK_DESCRIPTION_TOPIC, self.task_callback, state_qos_profile)
        self.joint_state_subscriber = self.create_subscription(JointState, STATE_TOPIC, self.joint_state_callback, reliable_qos_profile)

        # --- Publisher ---
        self.joint_command_publisher = self.create_publisher(JointState, ACTION_TOPIC, reliable_qos_profile)

        # --- Timer ---
        self.processing_timer = self.create_timer(PROCESSING_TIMER_PERIOD, self.process_and_publish_callback)
        self.get_logger().info(f"Processing timer set to {PROCESSING_TIMER_PERIOD} seconds.")

        self.get_logger().info(f"{NODE_NAME} initialization complete.")

    # --- Callback Functions ---
    def image_callback(self, msg: Image):
        self.latest_image_msg = msg

    def task_callback(self, msg: String):
        if self.latest_task_description_msg is None or self.latest_task_description_msg.data != msg.data:
             self.get_logger().info(f"Received new task description: '{msg.data}'")
             self.latest_task_description_msg = msg

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_states_msg = msg
        if not self.joint_names and msg.name:
            if len(msg.name) >= EXPECTED_JOINT_COUNT:
                self.joint_names = list(msg.name)[:EXPECTED_JOINT_COUNT] # Use only expected joints
                self.get_logger().info(f"Received and using joint names: {self.joint_names}")
                # Verify order if possible (e.g., against known names)
            else:
                self.get_logger().warn(f"Received {len(msg.name)} joints, expected {EXPECTED_JOINT_COUNT}." \
                                       " Cannot initialize joint names yet.")


    # --- Processing and Publishing Logic ---
    def process_and_publish_callback(self):
        """
        Timer callback: gets data, calls inference, publishes commands.
        """
        current_time = self.get_clock().now()

        # --- Check if all necessary data is available ---
        if (self.latest_image_msg is None or
            self.latest_task_description_msg is None or
            self.latest_joint_states_msg is None or
                not self.joint_names):
            # Throttle warning to avoid flooding logs while waiting
            if (current_time - self.last_command_time).nanoseconds > 5e9: # Log every 5 seconds
                 self.get_logger().warn("Waiting for image, task, or joint_states...",
                                        throttle_duration_sec=5.0, skip_first=False)
            return

        # --- Prepare Data for Inference ---
        try:
            # Convert Image using CV Bridge
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return
        except Exception as e:
             self.get_logger().error(f"Error converting image message: {e}")
             return

        try:
            # Extract Joint Positions (ensure correct number of joints)
            if len(self.latest_joint_states_msg.position) < EXPECTED_JOINT_COUNT:
                self.get_logger().warn(f"Joint states message has only {len(self.latest_joint_states_msg.position)}" \
                                       f" positions, expected {EXPECTED_JOINT_COUNT}. Skipping inference.")
                return
            # Use only the positions corresponding to self.joint_names
            current_joint_positions_np = np.array(self.latest_joint_states_msg.position[:EXPECTED_JOINT_COUNT])

            # Get Task Description
            task_description_str = self.latest_task_description_msg.data

            # Format observation dictionary using the client's helper method
            obs_dict = self.gr00t_client.format_observation(
                cv_image, current_joint_positions_np, task_description_str
            )

        except ValueError as e:
             self.get_logger().error(f"Value error during observation formatting: {e}")
             return
        except Exception as e:
            self.get_logger().error(f"Error preparing data for inference: {e}")
            return

        # --- Call Inference Server ---
        try:
            raw_action_chunk = self.gr00t_client.get_action(obs_dict)
        except Exception as e:
            self.get_logger().error(f"Error during inference server call: {e}")
            # Consider adding resilience: retry logic, fallback behavior?
            return

        # --- Extract Joint Commands from Response ---
        next_joint_positions = self.gr00t_client.extract_joint_commands(raw_action_chunk)

        if next_joint_positions is None:
            self.get_logger().warn("Failed to extract valid joint commands from inference result. Skipping command publishing.")
            return

        for joint_actions in next_joint_positions:
            # --- Construct and Publish Jointstate Message ---
            joint_cmd_msg = JointState()
            joint_cmd_msg.header.stamp = self.get_clock().now().to_msg()
            joint_cmd_msg.name = self.joint_names
            joint_cmd_msg.position = joint_actions.tolist()

            try:
                self.joint_command_publisher.publish(joint_cmd_msg)
                # self.get_logger().info(f"Published joint command: {list(joint_actions)}") # Debug
                self.last_command_time = current_time # Update time of last successful command
            except Exception as e:
                self.get_logger().error(f"Failed to publish joint command: {e}")
            
            # 20 hz    
            sleep(0.05)
            


        


def main(args=None):
    rclpy.init(args=args)
    robot_controller_node = None
    try:
        robot_controller_node = RobotControllerNode()
        rclpy.spin(robot_controller_node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"Unhandled exception during node execution/spin: {e}")
    finally:
        if robot_controller_node:
            robot_controller_node.destroy_node()
            print(f"Node {NODE_NAME} destroyed.")
        if rclpy.ok():
            rclpy.shutdown()
            print("rclpy shut down.")

if __name__ == '__main__':
    main()