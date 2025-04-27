# --- Imports ---
import os
import json
import jsonlines
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import time
import imageio
import shutil
import warnings
from typing import List, Dict, Tuple, Any, Optional, Set
import glob # Added for finding bag files

# ROS Bag related imports
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys.stores.ros2_humble import std_msgs__msg__Header
from rosbags.typesys.stores.ros2_humble import sensor_msgs__msg__Image as RosImage
from rosbags.typesys.stores.ros2_humble import sensor_msgs__msg__JointState as RosJointState
from rosbags.typesys.stores.ros2_humble import std_msgs__msg__String as RosString


# --- Configuration (MODIFY THESE FOR YOUR ROBOT AND DATA) ---

# Path to your ROS bag file or directory
BAG_INPUT_PATH = '/home/konu/Documents/groot/ros2bags/'
# Where to save the formatted dataset
OUTPUT_DATASET_DIR = './isaac_groot_custom_dataset' # Changed output dir name slightly
# A descriptive name for your dataset
DATASET_NAME = "so_100_isaacsim" # Used in info.json
# Specify the robot type (e.g., 'so100', 'franka', 'ur5')
ROBOT_TYPE = "so100"

# --- ROS Topics ---
# Add all relevant topics for your robot
IMAGE_TOPIC = '/rgb'
TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'
# *** State topic MUST be set for dynamic joint discovery ***
STATE_TOPIC = '/joint_states'
# *** Define the action topic ***
ACTION_TOPIC = '/joint_command' # Set your action topic here
# *** Define the task description topic ***
TASK_DESCRIPTION_TOPIC = '/task' # Topic for task description string

# --- TF Frames ---
# Set the correct frames for your setup
IMAGE_FRAME = 'Camera_link' # Frame ID of the sensor publishing images (used for video name)
WORLD_FRAME = 'Base' # World/reference frame for target pose calculations (if used)
# BASE_FRAME = 'base_link' # Robot base frame (if needed for TF lookups)

# --- Video Configuration ---
# Key for the main camera in modality.json and video folder name
VIDEO_KEY = "webcam" # Matches the example 'observation.images.webcam'

# --- Modality Definitions (CRITICAL - Match your actual data) ---
# State modalities for joints are now discovered automatically from STATE_TOPIC.
# You can still define OTHER state modalities here (e.g., for TF poses).
STATE_MODALITIES = {
    # Example: If you ALSO want End-Effector Pose from TF (in addition to discovered joints)
    # "ee_pose": {"start": <auto_detected_joint_dim>, "end": <auto_detected_joint_dim> + 7}
    # The start/end indices for non-joint states would need careful management
    # relative to the discovered joint dimension. For simplicity, focusing only on joints now.
}

# Update ACTION definition (if using actions)
ACTION_MODALITIES = {
    # Example: 6-DoF arm + 1-DoF gripper command
    # "arm_command": {"start": 0, "end": 5},    # Example
    # "gripper_command": {"start": 5, "end": 6} # Example
}


ANNOTATION_MODALITIES = {
    "human.task_description": {"original_key": "task_index"},
}
ANNOTATION_KEY_TO_TASK_INDEX = {}
DEFAULT_TASK_INDEX = 0

# --- Statistics Calculation ---
COMPUTE_STATS = True

# --- Internal Constants (Usually No Need to Change) ---
CODEBASE_VERSION = "v2.1"
DEFAULT_FPS = 30
SPLITS = {"train": "0:1"}
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# --- Helper Classes ---
class SimpleLogger:
    """A basic logger class."""
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARN: {msg}")
    def error(self, msg, exc_info=False):
        print(f"ERROR: {msg}", flush=True)
        if exc_info:
            import traceback
            traceback.print_exc()

logger = SimpleLogger()

# --- ROS Data Processing Class ---
class RosBagProcessor:
    """
    Handles reading ROS bag files and extracting relevant data.
    Automatically discovers joint state information if state_topic is provided.
    Processes action commands from the specified action_topic.
    Extracts task description from task_description_topic.
    """

    def __init__(self, image_topic: str, tf_topic: str, tf_static_topic: str,
                 state_topic: Optional[str] = None, action_topic: Optional[str] = None,
                 task_description_topic: Optional[str] = None,
                 logger: SimpleLogger = logger):
        """
        Initializes the RosBagProcessor.

        Args:
            bag_path: Path to the ROS bag file or directory.
            image_topic: The topic name for image messages.
            tf_topic: The topic name for dynamic transforms.
            tf_static_topic: The topic name for static transforms.
            state_topic: topic name for state messages
            action_topic: topic name for action messages.
            task_description_topic: topic name for task description messages.
            logger: Logger instance.
        """
        # self.bag_path = bag_path # REMOVED - path passed to process_bag
        self.image_topic = image_topic
        self.tf_topic = tf_topic
        self.tf_static_topic = tf_static_topic
        self.state_topic = state_topic
        self.action_topic = action_topic
        self.task_description_topic = task_description_topic
        self.logger = logger

        # Reset per-bag processing state
        self._reset_state()

        # State discovery attributes
        self.discovered_state_joint_names: Optional[List[str]] = None
        self.state_dim: int = 0 # Will be set after discovering joints
        
        # State discovery attributes
        self.discovered_action_joint_names: Optional[List[str]] = None
        self.action_dim: int = 0 # Will be set after discovering joints
        
        self._joint_discovery_done: bool = False


        if self.state_topic is None:
             self.logger.warning("state_topic is not provided. Cannot dynamically discover joint states.")
             self.state_dim = max(m['end'] for m in STATE_MODALITIES.values()) if STATE_MODALITIES else 0
        if self.action_topic is None:
             self.logger.warning("action_topic is not provided. Cannot dynamically discover joint states.")
             self.action_dim = max(m['end'] for m in ACTION_MODALITIES.values()) if ACTION_MODALITIES else 0
        if self.task_description_topic is None:
             self.logger.warning("task_description_topic is not provided. Cannot extract task descriptions.")


    def _reset_state(self):
        """Resets variables that change per bag file."""
        self.static_transforms = {}
        self.latest_dynamic_transforms = {}
        self.latest_state_msg: Optional[RosJointState] = None
        self.latest_action_msg: Optional[RosJointState] = None
        # Don't reset discovered joint names/dims here, they should be consistent

    def _discover_joint_states(self, reader: Rosbag2Reader) -> bool: # Renamed slightly
        """Scan the beginning of the bag for the first state/action message to get joint names."""
        # --- State Discovery ---
        if not self.state_topic:
            self.logger.warning("Cannot discover joint states: state_topic not set.")
            return False
        if not self.action_topic:
            self.logger.warning("Cannot discover joint states: action_topic not set.")
            return False

        self.logger.info(f"Scanning bag for first message on state topic: {self.state_topic}")
        state_connections = [c for c in reader.connections if c.topic == self.state_topic]
        if not state_connections:
            self.logger.error(f"State topic '{self.state_topic}' not found in bag connections.")
            return False

        found = False
        try:
            for connection, timestamp_ns, rawdata in reader.messages(connections=state_connections):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                if hasattr(msg, 'name') and hasattr(msg, 'position'):
                    self.discovered_state_joint_names = list(msg.name)
                    self.state_dim = len(self.discovered_state_joint_names)
                    self._joint_discovery_done = True
                    self.logger.info(f"Discovered {self.state_dim} joints from topic '{self.state_topic}': {self.discovered_state_joint_names}")
                    found = True
                    break # Found the first message
                else:
                    self.logger.warning(f"Received message on {self.state_topic} without 'name' or 'position' attributes. Skipping.")

        except Exception as e:
             self.logger.error(f"Error during joint state discovery scan: {e}", exc_info=True)
             return False

        if not found:
             self.logger.error(f"Failed to find a valid message on state topic '{self.state_topic}' to discover joint names.")
             return False
        

        self.logger.info(f"Scanning bag for first message on state topic: {self.action_topic}")
        state_connections = [c for c in reader.connections if c.topic == self.action_topic]
        if not state_connections:
            self.logger.error(f"State topic '{self.action_topic}' not found in bag connections.")
            return False

        found = False
        try:
            for connection, timestamp_ns, rawdata in reader.messages(connections=state_connections):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                if hasattr(msg, 'name') and hasattr(msg, 'position'):
                    self.discovered_action_joint_names = list(msg.name)
                    self.action_dim = len(self.discovered_action_joint_names)
                    self._joint_discovery_done = True
                    self.logger.info(f"Discovered {self.action_dim} joints from topic '{self.action_topic}': {self.discovered_action_joint_names}")
                    found = True
                    break # Found the first message
                else:
                    self.logger.warning(f"Received message on {self.action_topic} without 'name' or 'position' attributes. Skipping.")

        except Exception as e:
             self.logger.error(f"Error during joint state discovery scan: {e}", exc_info=True)
             return False

        if not found:
             self.logger.error(f"Failed to find a valid message on state topic '{self.action_topic}' to discover joint names.")
             return False

        return True

    def _extract_task_description(self, reader: Rosbag2Reader) -> str:
        """Reads the first message from the task description topic."""
        default_task = "Unknown Task"
        if not self.task_description_topic:
            self.logger.warning("Task description topic not set. Using default.")
            return default_task

        task_connections = [c for c in reader.connections if c.topic == self.task_description_topic]
        if not task_connections:
            self.logger.warning(f"Task description topic '{self.task_description_topic}' not found in bag. Using default.")
            return default_task

        try:
            for connection, _, rawdata in reader.messages(connections=task_connections):
                # Assuming std_msgs/String type
                if connection.msgtype == 'std_msgs/msg/String':
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    task_desc = msg.data
                    self.logger.info(f"Extracted task description: '{task_desc}'")
                    return task_desc
                else:
                    self.logger.warning(f"Unexpected message type '{connection.msgtype}' on task topic. Expected 'std_msgs/msg/String'. Skipping.")
                    # Continue searching in case the correct type appears later (unlikely for first msg)

        except Exception as e:
            self.logger.error(f"Error reading task description topic: {e}", exc_info=True)

        self.logger.warning(f"Could not extract valid task description from topic '{self.task_description_topic}'. Using default.")
        return default_task


    def _ros_image_to_pil(self, ros_image_msg: RosImage) -> Optional[Image.Image]:
        """Converts a ROS Image message to a PIL Image."""
        # [Keep implementation as before]
        try:
            encoding = ros_image_msg.encoding.lower()
            if encoding == 'rgb8':
                return Image.frombytes('RGB', (ros_image_msg.width, ros_image_msg.height), ros_image_msg.data.tobytes())
            elif encoding == 'bgr8':
                return Image.frombytes('RGB', (ros_image_msg.width, ros_image_msg.height), ros_image_msg.data.tobytes(), 'raw', 'BGR')
            elif encoding == 'mono8':
                return Image.frombytes('L', (ros_image_msg.width, ros_image_msg.height), ros_image_msg.data.tobytes())
            elif encoding == 'rgba8':
                 return Image.frombytes('RGBA', (ros_image_msg.width, ros_image_msg.height), ros_image_msg.data.tobytes())
            elif encoding == 'bgra8':
                 return Image.frombytes('RGBA', (ros_image_msg.width, ros_image_msg.height), ros_image_msg.data.tobytes(), 'raw', 'BGRA')
            else:
                self.logger.error(f"Unsupported image encoding: {ros_image_msg.encoding}")
                return None
        except Exception as e:
            self.logger.error(f"Error converting image: {e}")
            return None

    def _extract_transform_pose(self, transform_stamped) -> np.ndarray:
        """Extracts position and quaternion from a TransformStamped message."""
        pos = transform_stamped.transform.translation
        quat = transform_stamped.transform.rotation
        return np.array([pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w], dtype=np.float32)

    def _assemble_state(self) -> np.ndarray:
        """
        Assembles the current state vector based on latest messages and discovered joint order.
        """
        # Initialize state vector with the discovered dimension
        current_state = np.zeros(self.state_dim, dtype=np.float32)

        # --- Populate discovered joint states ---
        if self.latest_state_msg and self.discovered_state_joint_names:
            try:
                # Create a mapping from the latest message
                latest_positions = {name: pos for name, pos in zip(self.latest_state_msg.name, self.latest_state_msg.position)}

                # Populate state vector according to the discovered order
                for i, name in enumerate(self.discovered_state_joint_names):
                    if name in latest_positions:
                        current_state[i] = np.float32(latest_positions[name])
                    else:
                        # This shouldn't happen if discovery used the same topic, but handle defensively
                        self.logger.warning(f"Joint '{name}' (discovered) not found in latest state message. Using 0.0.")
                        current_state[i] = 0.0

            except AttributeError:
                self.logger.error("Could not access 'name' or 'position' in latest state message. Check message structure.")
            except Exception as e:
                self.logger.error(f"Error processing joint states: {e}")

        return current_state

    def _assemble_action(self) -> np.ndarray:
        """
        Assembles the current action vector based on the latest action message.
        """
        # Initialize action vector with the discovered dimension
        current_action = np.zeros(self.action_dim, dtype=np.float32)

        # --- Populate discovered joint states ---
        if self.latest_action_msg and self.discovered_action_joint_names:
            try:
                # Create a mapping from the latest message
                latest_positions = {name: pos for name, pos in zip(self.latest_action_msg.name, self.latest_action_msg.position)}

                # Populate action vector according to the discovered order
                for i, name in enumerate(self.discovered_action_joint_names):
                    if name in latest_positions:
                        current_action[i] = np.float32(latest_positions[name])
                    else:
                        # This shouldn't happen if discovery used the same topic, but handle defensively
                        self.logger.warning(f"Joint '{name}' (discovered) not found in latest action message. Using 0.0.")
                        current_action[i] = 0.0

            except AttributeError:
                self.logger.error("Could not access 'name' or 'position' in latest action message. Check message structure.")
            except Exception as e:
                self.logger.error(f"Error processing joint states: {e}")

        return current_action


    def process_bag(self, bag_path: str) -> Tuple[List[Dict[str, Any]], Dict, float, str]: 
        """
        Reads a single ROS bag, discovers/validates joint states, extracts task, and processes messages.

        Args:
            bag_path: Path to the specific ROS bag file or directory to process.

        Returns:
            Tuple containing:
            - episode_data: List of dictionaries for each step in this bag.
            - video_info: Dictionary with video dimensions for this bag.
            - fps: Calculated frames per second for this bag.
            - task_description: Extracted task description string for this bag.
            Returns ([], {}, DEFAULT_FPS, "Error Task") on failure.

        Raises:
            RuntimeError: If joint discovery/validation fails.
            ValueError: If required topics (like image) are missing.
            FileNotFoundError: If the bag_path doesn't exist.
        """
        self._reset_state() # Ensure clean state for this bag
        episode_data = []
        all_timestamps = []
        image_count = 0
        video_info = {"width": None, "height": None, "channels": None}
        task_description = "Error Task" # Default in case of early failure

        self.logger.info(f"--- Starting ROS bag processing: {bag_path} ---")

        try:
            with Rosbag2Reader(bag_path) as reader:
                # --- Discover Joint States First ---
                if not self._joint_discovery_done:
                    if not self._discover_joint_states(reader):
                         raise RuntimeError("Failed to discover joint states from the bag.")
                
                self.logger.info(f"Discovered State Dim: {self.state_dim}, Action Dim: {self.action_dim}")
            
            # opening reader again so that it starts from 0 frame
            with Rosbag2Reader(bag_path) as reader:
                # --- Main Processing Loop ---
                
                # --- Extract Task Description ---
                task_description = self._extract_task_description(reader)
                
                topics_to_read = [self.image_topic, self.tf_topic, self.tf_static_topic]
                if self.state_topic: topics_to_read.append(self.state_topic)
                if self.action_topic: topics_to_read.append(self.action_topic)
                connections = [c for c in reader.connections if c.topic in topics_to_read]

                if not any(c.topic == self.image_topic for c in connections):
                     raise ValueError(f"Image topic '{self.image_topic}' not found in bag.")

                self.logger.info(f"Processing messages from topics: {[c.topic for c in connections]}")

                # Use total message count for progress bar if possible
                pbar_desc = f"Processing Bag ({os.path.basename(bag_path)})"

                for connection, timestamp_ns, rawdata in tqdm(reader.messages(connections=connections), desc=pbar_desc):
                    try:
                        msg = deserialize_cdr(rawdata, connection.msgtype)
                        timestamp_sec = timestamp_ns / 1e9

                        # --- Process TF/TF_Static ---
                        if connection.topic == self.tf_static_topic:
                            for transform_stamped in msg.transforms:
                                child_frame = transform_stamped.child_frame_id.lstrip('/')
                                parent_frame = transform_stamped.header.frame_id.lstrip('/')
                                self.static_transforms[(child_frame, parent_frame)] = transform_stamped
                        elif connection.topic == self.tf_topic:
                            for transform_stamped in msg.transforms:
                                child_frame = transform_stamped.child_frame_id.lstrip('/')
                                parent_frame = transform_stamped.header.frame_id.lstrip('/')
                                self.latest_dynamic_transforms[(child_frame, parent_frame)] = transform_stamped

                        # --- Process State/Action Topics ---
                        elif connection.topic == self.state_topic:
                             self.latest_state_msg = msg # Update latest state
                        elif connection.topic == self.action_topic:
                             self.latest_action_msg = msg

                        # --- Process Image Messages (Triggers step creation) ---
                        elif connection.topic == self.image_topic:
                            # Ensure we have received at least one state message before processing images
                            if not self.latest_state_msg and self.state_topic:
                                # self.logger.warning(f"Skipping image at {timestamp_sec:.3f}s: No state message received yet.")
                                continue # Skip frame if state isn't available yet

                            image_count += 1
                            pil_image = self._ros_image_to_pil(msg)
                            if pil_image is None: continue

                            all_timestamps.append(timestamp_sec) # Collect timestamps for FPS calc

                            # Capture video dimensions from first frame
                            if video_info["height"] is None:
                                video_info["width"], video_info["height"] = pil_image.size
                                mode_to_channels = {'L': 1, 'RGB': 3, 'RGBA': 4, 'BGR': 3, 'BGRA': 4}
                                video_info["channels"] = mode_to_channels.get(pil_image.mode, 3)
                                if video_info["channels"] == 3 and pil_image.mode not in ['RGB', 'BGR']:
                                     self.logger.warning(f"Unknown image mode {pil_image.mode}. Assuming 3 channels.")
                                self.logger.info(f"Detected video dimensions: W={video_info['width']}, H={video_info['height']}, C={video_info['channels']}")

                            # --- Assemble State and Action ---
                            current_state = self._assemble_state()
                            current_action = self._assemble_action()

                            # --- Store Step Data ---
                            step_data = {
                                "timestamp": np.float32(timestamp_sec),
                                "observation.state": current_state,
                                "action": current_action,
                                "pil_image": pil_image, # Keep image temporarily
                            }
                            episode_data.append(step_data)

                    except Exception as e:
                         self.logger.error(f"Error processing message at time {timestamp_ns / 1e9:.3f} from topic {connection.topic}", exc_info=True)

        except FileNotFoundError:
            self.logger.error(f"ERROR: Bag file/directory not found at {bag_path}")
            raise
        except (RuntimeError, ValueError) as e: # Catch specific errors raised earlier
             self.logger.error(f"ERROR during bag processing for {bag_path}: {e}")
             return [], {}, DEFAULT_FPS, task_description # Return empty data but potentially valid task
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during bag processing for {bag_path}", exc_info=True)
            # Don't raise here, allow main loop to potentially skip this bag
            return [], {}, DEFAULT_FPS, task_description # Return empty data

        self.logger.info(f"Finished reading bag {bag_path}. Found {len(episode_data)} image frames.")

        # Calculate FPS
        fps = DEFAULT_FPS
        if len(all_timestamps) > 1:
            avg_diff = np.mean(np.diff(all_timestamps))
            if avg_diff > 1e-6: fps = 1.0 / avg_diff
            self.logger.info(f"Calculated FPS for this episode: {fps:.2f}")
        else:
            self.logger.warning(f"Not enough timestamps ({len(all_timestamps)}) to calculate FPS for this episode. Using default {DEFAULT_FPS}.")

        # Return processed data along with discovered state info
        if not self.discovered_state_joint_names:
             self.discovered_state_joint_names = [] # Ensure it's a list even if discovery failed
        # Return processed data along with discovered state info
        if not self.discovered_action_joint_names:
             self.discovered_action_joint_names = [] # Ensure it's a list even if discovery failed

        return episode_data, video_info, fps, task_description


# --- Dataset Formatting Class ---
class DatasetFormatter:
    """Handles formatting extracted data into the LeRobot dataset structure."""

    def __init__(self, output_dir: str, video_key: str,
                 state_dim: int, state_dim_names: List[str],
                 action_dim: int, action_dim_names: List[str],
                 logger: SimpleLogger = logger):
        """
        Initializes the DatasetFormatter.

        Args:
            output_dir: The root directory to save the dataset.
            video_key: The key used for the video modality (e.g., 'webcam').
            state_dim: Dimension of the state vector.
            state_dim_names: List of names for each state dimension.
            action_dim: Dimension of the action vector.
            action_dim_names: List of names for each action dimension.
            logger: Logger instance.
        """
        self.output_dir = output_dir
        self.video_key = video_key
        self.logger = logger

        # Store dimensions and names passed from processor
        self.state_dim = state_dim
        self.state_dim_names = state_dim_names
        self.action_dim = action_dim
        self.action_dim_names = action_dim_names

        # Define paths
        self.meta_dir = os.path.join(self.output_dir, 'meta')
        self.data_dir = os.path.join(self.output_dir, 'data', 'chunk-000')
        self.video_dir_base = os.path.join(self.output_dir, 'videos', 'chunk-000')
        self.video_dir_specific = os.path.join(self.video_dir_base, f'observation.images.{self.video_key}')

        # Validate names against dims (optional sanity check)
        if len(self.state_dim_names) != self.state_dim:
             logger.warning(f"Provided state_dim_names length ({len(self.state_dim_names)}) != state_dim ({self.state_dim}).")
        if len(self.action_dim_names) != self.action_dim:
             logger.warning(f"Provided action_dim_names length ({len(self.action_dim_names)}) != action_dim ({self.action_dim}).")

        # Prepare directories once
        self._ensure_dir(self.meta_dir)
        self._ensure_dir(self.data_dir)
        self._ensure_dir(self.video_dir_base)
        self._ensure_dir(self.video_dir_specific)

    def _ensure_dir(self, path: str):
        """Creates a directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    def _calculate_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculates statistics for numerical columns in the DataFrame."""
        # [Keep implementation as before]
        stats = {}
        self.logger.info(f"Calculating statistics for episode {df['episode_index'].iloc[0]}...")
        cols_to_stat = []
        if 'observation.state' in df.columns: cols_to_stat.append('observation.state')
        if 'action' in df.columns: cols_to_stat.append('action')
        if 'timestamp' in df.columns: cols_to_stat.append('timestamp')
        for col in ['frame_index', 'episode_index', 'index', 'task_index', 'reward', 'next.reward']:
             if col in df.columns:
                 cols_to_stat.append(col)
        # Include annotation columns if they exist and are numeric
        for col in df.columns:
             if col.startswith("annotation."):
                 cols_to_stat.append(col)

        for col_name in tqdm(cols_to_stat, desc="Calculating Stats"):
            try:
                if isinstance(df[col_name].iloc[0], (np.ndarray, list)):
                    data = np.stack(df[col_name].to_numpy()).astype(np.float64)
                    data[np.isinf(data)] = np.nan
                else:
                    data = df[col_name].to_numpy(dtype=np.float64)

                if np.isnan(data).any():
                     self.logger.warning(f"NaN values found in column '{col_name}'. Stats might be affected.")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    stats[col_name] = {
                        "mean": np.nanmean(data, axis=0).tolist(),
                        "std": np.nanstd(data, axis=0).tolist(),
                        "min": np.nanmin(data, axis=0).tolist(),
                        "max": np.nanmax(data, axis=0).tolist(),
                        "q01": np.nanpercentile(data, 1, axis=0).tolist(),
                        "q99": np.nanpercentile(data, 99, axis=0).tolist(),
                    }
            except Exception as e:
                self.logger.error(f"Could not calculate stats for column '{col_name}': {e}")
        self.logger.info("Statistics calculation finished.")
        return stats


    def write_metadata(self, all_episode_details: List[Dict], unique_tasks: Dict[str, int],
                       fps: float, video_info: Dict, last_episode_stats: Optional[Dict] = None):
        """
        Generates and writes all metadata files AFTER processing all bags.

        Args:
            all_episode_details: List of dicts, each {'index': int, 'length': int, 'tasks': List[str]}.
            unique_tasks: Dictionary mapping task description string to task_index.
            fps: Average or representative FPS (using last bag's FPS here).
            video_info: Video dimensions (using last bag's info here, assumed consistent).
            last_episode_stats: Optional stats dictionary from the last processed episode.
        """
        self.logger.info("Generating final metadata files...")

        total_episodes = len(all_episode_details)
        total_frames = sum(ep['length'] for ep in all_episode_details)
        total_tasks = len(unique_tasks)

        # 1. meta/modality.json
        _state_modalities = {}
        if len(STATE_MODALITIES) == 0:
             _state_modalities["single_arm"] = {"start": 0, "end": self.state_dim - 1}
             _state_modalities["gripper"] = {"start": self.state_dim - 1, "end": self.state_dim}
        else:
            _state_modalities = STATE_MODALITIES
            
        _action_modalities = {}
        if len(ACTION_MODALITIES) == 0:
             _action_modalities["single_arm"] = {"start": 0, "end": self.state_dim - 1}
             _action_modalities["gripper"] = {"start": self.state_dim - 1, "end": self.state_dim}
        else:
            _action_modalities = ACTION_MODALITIES

        modality_config = {
            "state": _state_modalities,
            "action": _action_modalities,
            "video": {
                self.video_key: {"original_key": f"observation.images.{self.video_key}"}
            },
        }
        if ANNOTATION_MODALITIES:
            modality_config["annotation"] = ANNOTATION_MODALITIES
            self.logger.info("Adding annotation modalities from config.")


        modality_file = os.path.join(self.meta_dir, 'modality.json')
        with open(modality_file, 'w') as f:
            json.dump(modality_config, f, indent=4)
        self.logger.info(f"Generated {modality_file}")

        # 2. meta/tasks.jsonl
        tasks_file = os.path.join(self.meta_dir, 'tasks.jsonl')
        with jsonlines.open(tasks_file, mode='w') as writer:
             # Sort tasks by index for consistent order
             sorted_tasks = sorted(unique_tasks.items(), key=lambda item: item[1])
             for task_desc, task_idx in sorted_tasks:
                 writer.write({"task_index": task_idx, "task": task_desc})
        self.logger.info(f"Generated {tasks_file} with {total_tasks} unique tasks.")


        # 3. meta/info.json
        info_config = {
            "codebase_version": CODEBASE_VERSION,
            "dataset_name": DATASET_NAME,
            "robot_type": ROBOT_TYPE,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_episodes,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": round(fps, 2),
            "splits": SPLITS,
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE,
            "features": {}
        }

        # Use instance dimensions and names
        if self.action_dim > 0:
            info_config["features"]["action"] = {
                "dtype": "float32",
                "shape": [self.action_dim],
                "names": self.action_dim_names
            }
        if self.state_dim > 0:
            info_config["features"]["observation.state"] = {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": self.state_dim_names # Use discovered names
            }
        if video_info["width"] is not None:
            video_feature_key = f"observation.images.{self.video_key}"
            info_config["features"][video_feature_key] = {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], video_info["channels"]],
                "names": ["height", "width", "channels"],
                "info": { "video.fps": round(fps, 2), "video.height": video_info["height"], "video.width": video_info["width"], "video.channels": video_info["channels"], "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False }
            }
        base_scalar_features = ["timestamp", "frame_index", "episode_index", "index", "task_index", "reward", "next.reward"]
        annotation_features = [f"annotation.{k}" for k in ANNOTATION_KEY_TO_TASK_INDEX.keys()]
        for col in base_scalar_features + annotation_features:
             dtype = "float32" if col  in ["timestamp", "reward", "next.reward"] else "int64"
             info_config["features"][col] = { "dtype": dtype, "shape": [1], "names": None }

        # Add boolean features
        for col in ["done", "next.done"]:
            info_config["features"][col] = { "dtype": "bool", "shape": [1], "names": None }


        info_file = os.path.join(self.meta_dir, 'info.json')
        with open(info_file, 'w') as f:
            json.dump(info_config, f, indent=4)
        self.logger.info(f"Generated {info_file}")

        # 4. meta/episodes.jsonl
        episodes_file = os.path.join(self.meta_dir, 'episodes.jsonl')
        with jsonlines.open(episodes_file, mode='w') as writer:
            # Sort by episode index just in case
            sorted_episodes = sorted(all_episode_details, key=lambda ep: ep['index'])
            for episode_detail in sorted_episodes:
                 writer.write({
                     "episode_index": episode_detail['index'],
                     "tasks": episode_detail['tasks'], # List of task descriptions for this episode
                     "length": episode_detail['length']
                 })
        self.logger.info(f"Generated {episodes_file} with {total_episodes} entries.")

        # 5. meta/stats.json (Optional, using last episode's stats)
        if COMPUTE_STATS and last_episode_stats:
            self._write_stats(last_episode_stats)
        elif COMPUTE_STATS:
             self.logger.warning("COMPUTE_STATS is True, but no stats data was provided for the last episode. Skipping stats.json.")


    def write_parquet(self, df: pd.DataFrame, episode_index: int):
        """Writes the episode data to a Parquet file."""
        # [Keep implementation as before]
        parquet_filename = f"episode_{episode_index:06d}.parquet"
        parquet_filepath = os.path.join(self.data_dir, parquet_filename)
        try:
            df.to_parquet(parquet_filepath, engine='pyarrow', index=False)
            self.logger.info(f"Generated {parquet_filepath}")
        except Exception as e:
            self.logger.error(f"Error writing Parquet file {parquet_filepath}: {e}", exc_info=True)
            raise


    def write_video(self, episode_frames: List[Image.Image], fps: float, episode_index: int):
        """Writes the episode frames to an MP4 video file."""
        video_filename = f"episode_{episode_index:06d}.mp4"
        video_filepath = os.path.join(self.video_dir_specific, video_filename)
        try:
            numpy_frames = []
            for frame in tqdm(episode_frames, desc="Converting frames for video"):
                 # Ensure RGB format for imageio
                 if frame.mode in ('RGBA', 'BGRA', 'P'): frame = frame.convert('RGB')
                 elif frame.mode == 'L': frame = frame.convert('RGB') # Convert grayscale to RGB
                 elif frame.mode == 'BGR':
                      # imageio expects RGB, so convert BGR
                      r, g, b = frame.split(); frame = Image.merge("RGB", (b, g, r))

                 # Check if conversion resulted in RGB
                 if frame.mode != 'RGB':
                      self.logger.warning(f"Unexpected frame mode '{frame.mode}'. Skipping.")
                      continue
                 numpy_frames.append(np.array(frame))

            if numpy_frames:
                imageio.mimwrite(video_filepath, numpy_frames, fps=fps, macro_block_size=16, quality=8)
                self.logger.info(f"Generated {video_filepath}")
            else:
                self.logger.warning("No frames collected/converted to write video.")
        except Exception as e:
            self.logger.error(f"Error writing video file {video_filepath}: {e}", exc_info=True)


    def _write_stats(self, stats_data: Dict):
        """Writes the calculated statistics to stats.json."""
        if stats_data:
            stats_file = os.path.join(self.meta_dir, 'stats.json')
            try:
                with open(stats_file, 'w') as f:
                    json.dump(stats_data, f, indent=4)
                self.logger.info(f"Generated {stats_file}")
            except Exception as e:
                self.logger.error(f"Error writing stats file: {e}")
        else:
            self.logger.warning("Stats calculation failed or produced no data. Skipping stats.json.")


# --- Main Execution Logic ---
def main():
    """Main function to run the conversion process for multiple bags."""
    start_time = time.time()

    # --- Find Bag Files ---
    bag_files = []
    if os.path.isdir(BAG_INPUT_PATH):
        # Find directories that contain metadata.yaml (indicator of a ROS2 bag)
        for root, dirs, files in os.walk(BAG_INPUT_PATH):
            if 'metadata.yaml' in files:
                bag_files.append(root)
                # Prevent os.walk from descending further into this directory
                dirs[:] = []
        logger.info(f"Found {len(bag_files)} potential bag directories in {BAG_INPUT_PATH}")
    elif os.path.isfile(BAG_INPUT_PATH) and BAG_INPUT_PATH.endswith('.db3'): # Simple check for single file (might need refinement)
         # Check if the directory containing the db3 also has metadata.yaml
         bag_dir = os.path.dirname(BAG_INPUT_PATH)
         if os.path.exists(os.path.join(bag_dir,'metadata.yaml')):
              bag_files.append(bag_dir)
              logger.info(f"Processing single bag directory: {bag_dir}")
         else:
              logger.warning(f"Input file {BAG_INPUT_PATH} provided, but its directory is missing metadata.yaml. Assuming it's the bag directory itself.")
              bag_files.append(BAG_INPUT_PATH) # Treat the path itself as the bag dir path
    elif os.path.exists(BAG_INPUT_PATH):
         # Assume it's a bag directory even without explicit check if it's not a file
         if os.path.exists(os.path.join(BAG_INPUT_PATH,'metadata.yaml')):
              bag_files.append(BAG_INPUT_PATH)
              logger.info(f"Processing single bag directory: {BAG_INPUT_PATH}")
         else:
              logger.error(f"Input path {BAG_INPUT_PATH} exists but doesn't seem to be a valid bag directory (missing metadata.yaml) or a directory containing bags.")
              exit(1)

    else:
        logger.error(f"BAG_INPUT_PATH '{BAG_INPUT_PATH}' does not exist.")
        exit(1)

    if not bag_files:
        logger.error(f"No ROS bag files found in '{BAG_INPUT_PATH}'.")
        exit(1)

    # --- Initialize Processors and Aggregators ---
    ros_processor = RosBagProcessor(
        image_topic=IMAGE_TOPIC,
        tf_topic=TF_TOPIC,
        tf_static_topic=TF_STATIC_TOPIC,
        state_topic=STATE_TOPIC,
        action_topic=ACTION_TOPIC,
        task_description_topic=TASK_DESCRIPTION_TOPIC, # Pass task topic
        logger=logger
    )

    dataset_formatter = None # Initialize later after getting dims from first bag
    all_episode_details = []
    unique_tasks: Dict[str, int] = {} # Map task string to index
    next_task_index = 0
    total_frames_processed = 0
    last_episode_stats = None
    first_bag_processed = False
    global_fps = DEFAULT_FPS # Use FPS from the first successfully processed bag
    global_video_info = {} # Use video info from the first successfully processed bag

    # --- Prepare Output Directory ---
    if os.path.exists(OUTPUT_DATASET_DIR):
        logger.warning(f"Removing existing output directory: {OUTPUT_DATASET_DIR}")
        shutil.rmtree(OUTPUT_DATASET_DIR)
    # Formatter will create meta dir; chunk dirs created during write

    # --- Process Each Bag ---
    for i, bag_path in enumerate(bag_files):
        episode_index = i # Start episode index from 0

        logger.info(f"\n>>> Processing Bag {i+1}/{len(bag_files)} (Episode {episode_index}): {bag_path}")

        try:
            # Process the current bag
            episode_data, video_info, fps, task_description = ros_processor.process_bag(bag_path)

            if not episode_data:
                logger.warning(f"Skipping bag {bag_path} due to processing errors or no image data found.")
                continue # Skip to the next bag

            # --- Initialize Formatter and Check Consistency (on first successful bag) ---
            if not first_bag_processed:
                # Use dimensions discovered from the first successful bag
                dataset_formatter = DatasetFormatter(
                    output_dir=OUTPUT_DATASET_DIR,
                    video_key=VIDEO_KEY,
                    state_dim=ros_processor.state_dim,
                    state_dim_names=ros_processor.discovered_state_joint_names,
                    action_dim=ros_processor.action_dim,
                    action_dim_names=ros_processor.discovered_action_joint_names,
                    logger=logger
                )
                global_fps = fps # Store FPS from first bag
                global_video_info = video_info # Store video info from first bag
                first_bag_processed = True
            else:
                # Sanity check dimensions against the formatter's stored dims
                if ros_processor.state_dim != dataset_formatter.state_dim or \
                   ros_processor.action_dim != dataset_formatter.action_dim:
                    logger.error(f"Inconsistent state/action dimensions in bag {bag_path}!")
                    logger.error(f"Expected State Dim: {dataset_formatter.state_dim}, Found: {ros_processor.state_dim}")
                    logger.error(f"Expected Action Dim: {dataset_formatter.action_dim}, Found: {ros_processor.action_dim}")
                    logger.warning(f"Skipping bag {bag_path} due to inconsistency.")
                    continue # Skip inconsistent bag
                # Could add checks for joint name consistency too if needed
                # Check video info consistency (optional but recommended)
                if video_info.get("width") != global_video_info.get("width") or \
                   video_info.get("height") != global_video_info.get("height") or \
                   video_info.get("channels") != global_video_info.get("channels"):
                    logger.warning(f"Inconsistent video dimensions detected in bag {bag_path}. Using dimensions from the first bag ({global_video_info}).")
                    # Proceed, but be aware metadata might not perfectly match all videos


            # --- Handle Task Description ---
            if task_description not in unique_tasks:
                unique_tasks[task_description] = next_task_index
                current_task_index = next_task_index
                next_task_index += 1
                logger.info(f"New task found: '{task_description}' (index: {current_task_index})")
            else:
                current_task_index = unique_tasks[task_description]

            # --- Prepare DataFrame for this Episode ---
            episode_frames_pil = [step_data.pop("pil_image") for step_data in episode_data]
            df = pd.DataFrame(episode_data)
            num_frames_this_episode = len(df)

            if num_frames_this_episode == 0:
                 logger.warning(f"No data frames created for episode {episode_index}. Skipping file writing.")
                 continue

            # Add standard LeRobot columns
            df['episode_index'] = np.int64(episode_index)
            df['frame_index'] = np.arange(num_frames_this_episode, dtype=np.int64)
            df['index'] = np.arange(total_frames_processed, total_frames_processed + num_frames_this_episode, dtype=np.int64) # Global index
            df['task_index'] = np.int64(current_task_index)
            for k, task_idx_val in ANNOTATION_KEY_TO_TASK_INDEX.items():
                df[f"annotation.{k}"] = np.int64(task_idx_val)
            df['reward'] = np.float32(0.0)
            df['done'] = False
            df.loc[df.index[-1], 'done'] = True # Mark last frame as done
            # Calculate next state columns (handle boundary)
            df['next.reward'] = df['reward'].shift(-1).fillna(0.0).astype(np.float32)
            df['next.done'] = df['done'].shift(-1).fillna(False).astype(bool)
            # Ensure next state/action are handled if needed (often not stored directly)

            # Define column order
            cols_order = []
            if dataset_formatter.action_dim > 0: cols_order.append("action")
            if dataset_formatter.state_dim > 0: cols_order.append("observation.state")
            cols_order.extend(["timestamp", "frame_index", "episode_index", "index", "task_index"])
            cols_order.extend([f"annotation.{k}" for k in ANNOTATION_KEY_TO_TASK_INDEX.keys()])
            cols_order.extend(["reward", "done", "next.reward", "next.done"])

            final_cols = [col for col in cols_order if col in df.columns]
            missing_cols = set(cols_order) - set(final_cols)
            if missing_cols:
                logger.warning(f"Columns missing from DataFrame for final ordering in ep {episode_index}: {missing_cols}")
            remaining_cols = [col for col in df.columns if col not in final_cols]
            final_cols.extend(remaining_cols)
            df = df[final_cols]

            # --- Write Episode Files ---
            dataset_formatter.write_parquet(df, episode_index)
            dataset_formatter.write_video(episode_frames_pil, fps, episode_index)

            # --- Calculate Stats (Optional) ---
            if COMPUTE_STATS:
                last_episode_stats = dataset_formatter._calculate_stats(df) # Overwrites previous stats

            # --- Aggregate Information ---
            all_episode_details.append({
                "index": episode_index,
                "length": num_frames_this_episode,
                "tasks": [task_description] # Store task description as a list
            })
            total_frames_processed += num_frames_this_episode

        except Exception as e:
            logger.error(f"Failed to process bag {bag_path} completely.", exc_info=True)
            logger.warning(f"Skipping bag {bag_path} due to unexpected error.")
            # Continue to the next bag

    # --- Finalization ---
    if not first_bag_processed or dataset_formatter is None:
         logger.error("No bags were processed successfully. Cannot generate metadata.")
         exit(1)

    # Write final metadata files using aggregated info
    logger.info("\n--- Writing Final Metadata ---")
    dataset_formatter.write_metadata(
        all_episode_details,
        unique_tasks,
        global_fps, # Use FPS from first bag
        global_video_info, # Use video info from first bag
        last_episode_stats if COMPUTE_STATS else None
    )

    processing_time = time.time() - start_time
    logger.info(f"\nDataset formatting complete for {len(all_episode_details)} episodes.")
    logger.info(f"Total frames processed: {total_frames_processed}")
    logger.info(f"Output directory: {OUTPUT_DATASET_DIR}")
    logger.info(f"Total execution time: {processing_time:.2f} seconds.")


if __name__ == "__main__":
    # --- Sanity Checks on Configuration ---
    # Modified check for input path existence
    if not BAG_INPUT_PATH:
        logger.error("BAG_INPUT_PATH is not set.")
        exit(1)
    if not OUTPUT_DATASET_DIR:
        logger.error("OUTPUT_DATASET_DIR is not set.")
        exit(1)
    if not IMAGE_TOPIC:
        logger.error("IMAGE_TOPIC is not set.")
        exit(1)
    if not STATE_TOPIC:
        logger.error("STATE_TOPIC must be set in the configuration for dynamic joint discovery.")
        exit(1)
    if not ACTION_TOPIC:
        logger.error("ACTION_TOPIC must be set in the configuration for dynamic joint discovery.")
        exit(1)
    if not TASK_DESCRIPTION_TOPIC:
        logger.warning("TASK_DESCRIPTION_TOPIC is not set. Task descriptions will be default/unknown.")
        exit(1)
    if not STATE_MODALITIES:
        logger.warning("STATE_MODALITIES is not defined in config, It will be dynamically determined for 'single_arm' from the state topic. If this behaviour is undesired, Update the variable or edit the modality.json later")
    if not ACTION_MODALITIES:
        logger.warning("ACTION_MODALITIES is not defined in config. It will be dynamically determined.")

    main()