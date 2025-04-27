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
from typing import List, Dict, Tuple, Any, Optional

# ROS Bag related imports
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys.stores.ros2_humble import std_msgs__msg__Header
from rosbags.typesys.stores.ros2_humble import sensor_msgs__msg__Image as RosImage
from rosbags.typesys.stores.ros2_humble import sensor_msgs__msg__JointState as RosJointState

# --- Configuration (MODIFY THESE FOR YOUR ROBOT AND DATA) ---

# Path to your ROS bag file or directory
BAG_FILE_PATH = '/home/konu/Documents/groot/ros2bags/so100_cup_side'
# Where to save the formatted dataset
OUTPUT_DATASET_DIR = './isaac_so100_groot_dataset_side' # Changed output dir name slightly
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

# --- Annotation Definitions ---
TASKS_DATA = [
    {"task_index": 0, "task": "Approach the cup."},
]
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
    """

    def __init__(self, bag_path: str, image_topic: str, tf_topic: str, tf_static_topic: str,
                 state_topic: Optional[str] = None, action_topic: Optional[str] = None,
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
            logger: Logger instance.
        """
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.tf_topic = tf_topic
        self.tf_static_topic = tf_static_topic
        self.state_topic = state_topic
        self.action_topic = action_topic
        self.logger = logger

        self.static_transforms = {}
        self.latest_dynamic_transforms = {}
        self.latest_state_msg: Optional[RosJointState] = None
        self.latest_action_msg: Optional[RosJointState] = None # Define type hint if action message type is known

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


    def _discover_joint_states(self, reader: Rosbag2Reader):
        """Scan the beginning of the bag for the first state message to get joint names."""
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
                latest_positions = {name: pos for name, pos in zip(self.latest_state_msg.name, self.latest_state_msg.position)}

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

    def process_bag(self) -> Tuple[List[Dict[str, Any]], Dict, float, int, List[str], int, List[str]]:
        """
        Reads the ROS bag, discovers joint states, and processes messages.

        Returns:
            Tuple containing:
            - episode_data: List of dictionaries for each step.
            - video_info: Dictionary with video dimensions.
            - fps: Calculated frames per second.
            - state_dim: Discovered state dimension (number of joints).
            - state_dim_names: Discovered list of state joint names.
            - action_dim: Discovered action dimension (number of joints).
            - action_dim_names: Discovered list of action joint names.
        """
        episode_data = []
        all_timestamps = []
        image_count = 0
        video_info = {"width": None, "height": None, "channels": None}

        self.logger.info(f"Starting ROS bag processing: {self.bag_path}")

        try:
            with Rosbag2Reader(self.bag_path) as reader:
                # --- Discover Joint States First ---
                if not self._joint_discovery_done:
                    if not self._discover_joint_states(reader):
                         raise RuntimeError("Failed to discover joint states from the bag.")
                
                self.logger.info(f"Discovered State Dim: {self.state_dim}, Action Dim: {self.action_dim}")
            
            # opening reader again so that it starts from 0 frame
            with Rosbag2Reader(self.bag_path) as reader:
                # --- Main Processing Loop ---
                topics_to_read = [self.image_topic, self.tf_topic, self.tf_static_topic]
                if self.state_topic: topics_to_read.append(self.state_topic)
                if self.action_topic: topics_to_read.append(self.action_topic)
                connections = [c for c in reader.connections if c.topic in topics_to_read]

                if not any(c.topic == self.image_topic for c in connections):
                     raise ValueError(f"Image topic '{self.image_topic}' not found in bag.")

                self.logger.info(f"Processing messages from topics: {[c.topic for c in connections]}")

                for connection, timestamp_ns, rawdata in tqdm(reader.messages(connections=connections), desc="Processing Bag"):
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
            self.logger.error(f"ERROR: Bag file/directory not found at {self.bag_path}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during bag processing", exc_info=True)
            raise

        self.logger.info(f"Finished reading bag. Found {len(episode_data)} image frames.")

        # Calculate FPS
        fps = DEFAULT_FPS
        if len(all_timestamps) > 1:
            avg_diff = np.mean(np.diff(all_timestamps))
            if avg_diff > 1e-6: fps = 1.0 / avg_diff
            self.logger.info(f"Calculated FPS: {fps:.2f}")
        else:
            self.logger.warning(f"Not enough timestamps ({len(all_timestamps)}) to calculate FPS. Using default {DEFAULT_FPS}.")

        # Return processed data along with discovered state info
        if not self.discovered_state_joint_names:
             self.discovered_state_joint_names = [] # Ensure it's a list even if discovery failed
        # Return processed data along with discovered state info
        if not self.discovered_action_joint_names:
             self.discovered_action_joint_names = [] # Ensure it's a list even if discovery failed

        return episode_data, video_info, fps, self.state_dim, self.discovered_state_joint_names, self.action_dim, self.discovered_action_joint_names


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


    def _ensure_dir(self, path: str):
        """Creates a directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    def _calculate_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculates statistics for numerical columns in the DataFrame."""
        # [Keep implementation as before]
        stats = {}
        self.logger.info("Calculating statistics...")
        cols_to_stat = []
        if 'observation.state' in df.columns: cols_to_stat.append('observation.state')
        if 'action' in df.columns: cols_to_stat.append('action')
        if 'timestamp' in df.columns: cols_to_stat.append('timestamp')
        for col in ['frame_index', 'episode_index', 'index', 'task_index'] + [f"annotation.{k}" for k in ANNOTATION_KEY_TO_TASK_INDEX.keys()]:
             if col in df.columns:
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


    def _write_metadata(self, total_frames: int, total_episodes: int, fps: float, video_info: Dict):
        """Generates and writes all metadata files using instance dimensions/names."""
        self.logger.info("Generating metadata files...")

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
            "state": _state_modalities, # Using only discovered joints for now
            "action": _action_modalities,
            "video": {
                self.video_key: {"original_key": f"observation.images.{self.video_key}"}
            },
        }
        if ANNOTATION_MODALITIES:
            modality_config["annotation"] = ANNOTATION_MODALITIES

        modality_file = os.path.join(self.meta_dir, 'modality.json')
        with open(modality_file, 'w') as f:
            json.dump(modality_config, f, indent=4)
        self.logger.info(f"Generated {modality_file}")

        # 2. meta/tasks.jsonl
        # [Keep implementation as before]
        tasks_file = os.path.join(self.meta_dir, 'tasks.jsonl')
        with jsonlines.open(tasks_file, mode='w') as writer:
            for task_data in TASKS_DATA:
                writer.write(task_data)
        self.logger.info(f"Generated {tasks_file}")


        # 3. meta/info.json
        info_config = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(TASKS_DATA),
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
        base_scalar_features = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        annotation_features = [f"annotation.{k}" for k in ANNOTATION_KEY_TO_TASK_INDEX.keys()]
        for col in base_scalar_features + annotation_features:
             dtype = "float32" if col == "timestamp" else "int64"
             info_config["features"][col] = { "dtype": dtype, "shape": [1], "names": None }

        info_file = os.path.join(self.meta_dir, 'info.json')
        with open(info_file, 'w') as f:
            json.dump(info_config, f, indent=4)
        self.logger.info(f"Generated {info_file}")

        # 6. meta/episodes.jsonl
        # [Keep implementation as before]
        episodes_file = os.path.join(self.meta_dir, 'episodes.jsonl')
        with jsonlines.open(episodes_file, mode='w') as writer:
            writer.write({ "episode_index": 0, "tasks": [data['task'] for data in TASKS_DATA], "length": total_frames })
        self.logger.info(f"Generated {episodes_file}")


    def _write_parquet(self, df: pd.DataFrame, episode_index: int):
        """Writes the episode data to a Parquet file."""
        # [Keep implementation as before]
        parquet_filename = f"episode_{episode_index:06d}.parquet"
        parquet_filepath = os.path.join(self.data_dir, parquet_filename)
        try:
            df.to_parquet(parquet_filepath, engine='pyarrow', index=False)
            self.logger.info(f"Generated {parquet_filepath}")
        except Exception as e:
            self.logger.error(f"Error writing Parquet file: {e}")
            raise


    def _write_video(self, episode_frames: List[Image.Image], fps: float, episode_index: int):
        """Writes the episode frames to an MP4 video file."""
        # [Keep implementation as before]
        video_filename = f"episode_{episode_index:06d}.mp4"
        video_filepath = os.path.join(self.video_dir_specific, video_filename)
        try:
            numpy_frames = []
            for frame in tqdm(episode_frames, desc="Converting frames for video"):
                 if frame.mode in ('RGBA', 'BGRA', 'P'): frame = frame.convert('RGB')
                 elif frame.mode == 'L': frame = frame.convert('RGB')
                 elif frame.mode == 'BGR':
                      r, g, b = frame.split(); frame = Image.merge("RGB", (b, g, r))
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
            self.logger.error(f"Error writing video file: {e}", exc_info=True)


    def _write_stats(self, stats_data: Dict):
        """Writes the calculated statistics to stats.json."""
        # [Keep implementation as before]
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


    def format_dataset(self, episode_data: List[Dict[str, Any]], video_info: Dict, fps: float):
        """
        Formats the processed data into the LeRobot dataset structure.
        """
        if not episode_data:
            self.logger.error("ERROR: No data provided to format. Aborting.")
            return

        total_frames = len(episode_data)
        episode_index = 0 # Assuming single episode
        self.logger.info(f"Formatting dataset with {total_frames} steps for episode {episode_index}.")

        # --- Prepare Directories ---
        if os.path.exists(self.output_dir):
            self.logger.warning(f"Removing existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self._ensure_dir(self.meta_dir)
        self._ensure_dir(self.data_dir)
        self._ensure_dir(self.video_dir_specific)

        # --- Prepare DataFrame ---
        episode_frames = [step_data.pop("pil_image") for step_data in episode_data]
        df = pd.DataFrame(episode_data)

        # Add standard LeRobot columns
        df['episode_index'] = np.int64(episode_index)
        df['frame_index'] = np.arange(total_frames, dtype=np.int64)
        df['index'] = np.arange(total_frames, dtype=np.int64)
        df['task_index'] = np.int64(DEFAULT_TASK_INDEX)
        for k, task_idx_val in ANNOTATION_KEY_TO_TASK_INDEX.items():
            df[f"annotation.{k}"] = np.int64(task_idx_val)
        df['reward'] = np.float32(0.0)
        df['done'] = False
        if not df.empty:
            df.loc[df.index[-1], 'done'] = True
        df['next.reward'] = df['reward'].shift(-1).fillna(0.0).astype(np.float32)
        df['next.done'] = df['done'].shift(-1).fillna(False).astype(bool)

        # Define column order
        cols_order = []
        if self.action_dim > 0: cols_order.append("action")
        if self.state_dim > 0: cols_order.append("observation.state")
        cols_order.extend(["timestamp", "frame_index", "episode_index", "index", "task_index"])
        cols_order.extend([f"annotation.{k}" for k in ANNOTATION_KEY_TO_TASK_INDEX.keys()])
        cols_order.extend(["next.reward", "next.done"])

        final_cols = [col for col in cols_order if col in df.columns]
        missing_cols = set(cols_order) - set(final_cols)
        if missing_cols:
            self.logger.warning(f"Columns missing from DataFrame for final ordering: {missing_cols}")
        remaining_cols = [col for col in df.columns if col not in final_cols]
        final_cols.extend(remaining_cols)
        df = df[final_cols]

        # --- Write Files ---
        self._write_parquet(df, episode_index)
        self._write_video(episode_frames, fps, episode_index)
        self._write_metadata(total_frames, 1, fps, video_info) # Assuming 1 episode

        # --- Calculate and Write Stats (Optional) ---
        if COMPUTE_STATS:
            stats_data = self._calculate_stats(df)
            self._write_stats(stats_data)
        else:
            self.logger.info("Skipping statistics calculation (COMPUTE_STATS=False).")

        self.logger.info("\nDataset formatting complete!")
        self.logger.info(f"Output directory: {self.output_dir}")


# --- Main Execution Logic ---
def main():
    """Main function to run the conversion process."""
    start_time = time.time()

    # 1. Initialize ROS Processor
    ros_processor = RosBagProcessor(
        bag_path=BAG_FILE_PATH,
        image_topic=IMAGE_TOPIC,
        tf_topic=TF_TOPIC,
        tf_static_topic=TF_STATIC_TOPIC,
        state_topic=STATE_TOPIC,
        action_topic=ACTION_TOPIC,
        logger=logger
    )

    # 2. Process ROS Bag Data (includes joint discovery)
    try:
        # process_bag now returns discovered state info
        episode_data, video_info, fps, discovered_state_dim, discovered_state_names, discovered_action_dim, discovered_action_names = ros_processor.process_bag()
    except Exception as e:
        logger.error(f"Failed to process ROS bag. Exiting.", exc_info=False)
        exit(1)

    # 3. Initialize Formatter *after* processing, with discovered info
    dataset_formatter = DatasetFormatter(
        output_dir=OUTPUT_DATASET_DIR,
        video_key=VIDEO_KEY,
        state_dim=discovered_state_dim,
        state_dim_names=discovered_state_names,
        action_dim=discovered_action_dim, # Get action dim from processor
        action_dim_names=discovered_action_names,   # Action names still from config
        logger=logger
    )

    # 4. Format and Save Dataset
    try:
        dataset_formatter.format_dataset(episode_data, video_info, fps)
    except Exception as e:
        logger.error(f"Failed to format dataset.", exc_info=True)
        exit(1)

    processing_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {processing_time:.2f} seconds.")

if __name__ == "__main__":
    # --- Sanity Checks on Configuration ---
    if not BAG_FILE_PATH or not os.path.exists(BAG_FILE_PATH):
        logger.error(f"BAG_FILE_PATH '{BAG_FILE_PATH}' is not set or does not exist.")
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
    if not STATE_MODALITIES:
        logger.warning("STATE_MODALITIES is not defined in config, It will be dynamically determined for 'single_arm' from the state topic. If this behaviour is undesired, Update the variable or edit the modality.json later")
    if not ACTION_MODALITIES:
        logger.warning("ACTION_MODALITIES is not defined in config. It will be dynamically determined.")

    main()
