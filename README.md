# 🤖 ROS2 Bag to Lerobot Dataset Converter 💾

This repository provides a tool (`ros2_convert.py`) to convert ROS2 bag files into the [Lerobot](https://github.com/huggingface/lerobot) dataset format. This allows you to easily process your robot's recorded data (joint states, actions, images, etc.) and prepare it for training manipulation policies, such as [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).

---

## 📋 Table of Contents

* [🚀 Installation](#-installation)
* [⚙️ Configuration](#️-configuration)
* [🏃‍♀️ Usage](#️-usage)
* [✨ Key Features & Notes](#-key-features--notes)
* [✅ Validation with Isaac Sim](#-validation-with-isaac-sim)
* [📄 License](#-license)

---

## 🚀 Installation

Before you begin, ensure you have the necessary prerequisites installed:

1.  **ROS2:** Install ROS2 (Humble, Iron, etc.) following the official documentation for your OS:
    * ➡️ [ROS2 Installation Guide](https://docs.ros.org/en/rolling/Installation.html)
    * *Remember to source your ROS2 environment in any terminal where you run ROS2 commands or this script.*

2.  **Lerobot & Dependencies:** Install `lerobot` and other required Python packages using pip:
    ```bash
    pip install lerobot datasets numpy Pillow huggingface_hub
    ```
    📦 *Note: Ensure `lerobot` is installed correctly. Refer to its official installation guide if needed.*

---

## ⚙️ Configuration

Before running the conversion script, you **must** update the configuration variables within the `ros2_convert.py` file to match your setup:

```python
# --- Core Configuration ---

# Path to your ROS bag file OR a directory containing multiple bag files
# Example: '/path/to/my_robot/bags/' or '/path/to/single/bagfile.db3'
BAG_INPUT_PATH = '/home/konu/Documents/groot/ros2bags/'

# Directory where the converted Lerobot dataset will be saved
OUTPUT_DATASET_DIR = './isaac_groot_custom_dataset'

# A descriptive name for your dataset (used in info.json)
DATASET_NAME = "so_100_isaacsim"

# Specify the robot type (used for potential robot-specific logic, e.g., 'so100', 'franka', 'ur5')
ROBOT_TYPE = "so100"

# --- ROS Topic Configuration ---
# Update these topics to match the topics in YOUR ROS bag file!

# Image topic (RGB recommended)
IMAGE_TOPIC = '/rgb'

# TF topics (used for coordinate transforms)
TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'

# State topic (MUST be set for dynamic joint discovery and state recording)
# This should typically be your '/joint_states' topic or equivalent
STATE_TOPIC = '/joint_states'

# Action topic (The topic your robot subscribes to for commands)
# Example: '/joint_command', '/arm_controller/joint_trajectory', etc.
ACTION_TOPIC = '/joint_command'

# ask description topic 
TASK_DESCRIPTION_TOPIC = '/task' # NEW: Topic for task description string

# --- TF Frames ---
# Set the correct frames for your setup
IMAGE_FRAME = 'Camera_link' # Frame ID of the sensor publishing images (used for video name)
WORLD_FRAME = 'Base' # World/reference frame for target pose calculations (if used)

# Key for the main camera in modality.json and video folder name
VIDEO_KEY = "webcam" # Matches the example 'observation.images.webcam'

# --- Modality Configuration (Advanced) ---
# Define how state and action joints are grouped.
# If left empty, the script will attempt automatic grouping (see notes below).
STATE_MODALITIES = {} # Example: {'arm': ['joint1', 'joint2', ...], 'gripper': ['gripper_joint']}
ACTION_MODALITIES = {} # Example: {'arm': ['joint1', 'joint2', ...], 'gripper': ['gripper_joint']}
```

📝 **Important:** Make sure the `BAG_FILE_PATH`, `OUTPUT_DATASET_DIR`, and all `_TOPIC` variables accurately reflect your ROS bag data and desired output location.

---

## 🏃‍♀️ Usage

1.  **Configure:** Modify the variables in `ros2_convert.py` as described in the [Configuration](#️-configuration) section.
2.  **Source ROS2:** Open a terminal and source your ROS2 environment:
    ```bash
    # Example for ROS2 Humble (replace with your ROS distro if different)
    source /opt/ros/humble/setup.bash
    # If you have a custom workspace overlay, source that too:
    # source install/setup.bash
    ```
3.  **Run the Script:** Navigate to the directory containing `ros2_convert.py` and execute it:
    ```bash
    python3 ros2_convert.py
    ```
4.  **Output:** The script will process the ROS bag(s) specified in `BAG_FILE_PATH` and save the resulting Lerobot dataset to the `OUTPUT_DATASET_DIR`.

🎉 You now have your own dataset ready for training with `lerobot` and potentially deploying with models like `grootn1`!

---

## ✨ Key Features & Notes

* **Image-Triggered Steps:** 🖼️ The dataset creation process triggers a new "step" only when a message arrives on the `IMAGE_TOPIC`. This ensures the dataset frequency matches your image data, assuming images are the slowest-published modality relevant for policy learning.
* **Automatic State Modality Grouping:** ⚙️ If `STATE_MODALITIES` is left empty (`{}`), the script attempts to automatically group joints based on the `STATE_TOPIC` (`/joint_states`). It assumes a standard setup where the last joint is the 'gripper' and the rest belong to the 'single_arm'.
* **Automatic Action Modality Grouping:** 🦾 Similarly, if `ACTION_MODALITIES` is empty (`{}`), the script mirrors the state grouping logic for actions, assuming the command structure matches the state structure (last joint = gripper, others = arm).
* **Prerequisites:** Your ROS2 bag **must** contain messages on the topics specified in the configuration, particularly:
    * `STATE_TOPIC` (e.g., `/joint_states`) for robot joint positions/velocities.
    * `ACTION_TOPIC` (e.g., `/joint_command`) for the commands sent to the robot.
    * `IMAGE_TOPIC` (e.g., `/rgb`) for visual observations.
    * `TF_TOPIC` / `TF_STATIC_TOPIC` (e.g., `/tf`, `/tf_static`) for coordinate frame information, if relevant.
    * `TASK_DESCRIPTION_TOPIC` (e.g., `/task`)` topic for task descriptions.

---

## ✅ Validation with Isaac Sim

This section details how to validate a policy (e.g., a fine-tuned `grootn1`) trained on the generated dataset using the provided Isaac Sim example environment.

An example setup for an `so-arm100` robot with ROS2 integration in Isaac Sim is provided in the `ros2_ws` directory.

1.  **Build the ROS2 Workspace:**
    * Navigate to the example workspace directory: `cd ros2_ws`
    * Install dependencies using `rosdep`:
        ```bash
        rosdep install --from-paths src --ignore-src -r -y
        ```
    * Build the workspace using `colcon`:
        ```bash
        colcon build --symlink-install
        ```
    * Source the newly built workspace:
        ```bash
        source install/setup.bash
        ```
    🛠️ *Remember to source your main ROS2 environment **before** building and sourcing the local workspace.*

2.  **Launch Isaac Sim Environment:** Start the Isaac Sim simulation with the `so-arm100` robot example. (Specific launch instructions depend on your Isaac Sim setup).

3.  **Run the Policy Bridge Node:**
    * Open a **new terminal** (ensure it's sourced with both ROS2 and your `ros2_ws`).
    * Run the example `grootn1` to ROS2 bridging node (replace `robot_controller` if your package/node names differ):
        ```bash
        ros2 run robot_controller robot_controller_node
        ```
    * This node will likely subscribe to a task topic and publish joint commands based on the policy's output.

4.  **Send a Task Command:**
    * Open **another new terminal** (sourced).
    * Publish a task description to the appropriate topic (defined in your controller node, e.g., `/task`):
        ```bash
        ros2 topic pub /task std_msgs/msg/String "data: 'Approach the cup.'"
        ```
    🎯 You should now see the robot in Isaac Sim attempt to execute the commanded task based on the loaded policy!

5.  **(Optional) MoveIt2 Demo:**
    * If you want to test the MoveIt2 setup included in the `ros2_ws` independently, run:
        ```bash
        # Ensure the workspace is sourced
        ros2 launch so_arm100_moveit_config demo.launch.py
        ```

---

## 📄 License

(Optional: Add your license information here. Common choices include MIT, Apache 2.0, GPL, etc.)

Example:
```
This project is licensed under the MIT License - see the LICENSE.md file for details.
```

---
