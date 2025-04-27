import os

import datasets
import numpy as np
from PIL import Image  # Example if you have image observations
from tqdm.auto import tqdm  # For progress bars

# --- 1. Prepare Your Raw Data (Example Placeholder) ---
# Assume you have data loaded into a structure like this:
# A list of episodes, where each episode is a list of timesteps.
# Each timestep contains observation, action, reward, etc.

# Let's create some DUMMY data for demonstration
raw_episodes = []
num_episodes = 5
max_steps_per_episode = 50

print("Generating dummy raw data...")
for ep_idx in range(num_episodes):
    episode = []
    # Dummy observation components
    img_height, img_width = 64, 64
    state_dim = 4
    action_dim = 3

    # Simulate varying episode lengths
    num_steps = np.random.randint(max_steps_per_episode // 2, max_steps_per_episode)

    for step_idx in range(num_steps):
        # --- Your actual data loading/access logic would go here ---
        dummy_image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
        dummy_state = np.random.rand(state_dim).astype(np.float32)
        dummy_action = np.random.randn(action_dim).astype(np.float32)
        dummy_reward = float(np.random.rand())
        is_terminal = (step_idx == num_steps - 1) # Example termination
        is_truncated = False # Example truncation (e.g., time limit)
        # --- End of dummy data generation for one step ---

        timestep_data = {
            "observation": {
                # Using PIL Image object for datasets.Image feature
                "image": Image.fromarray(dummy_image),
                "state": dummy_state,
            },
            "action": dummy_action,
            "reward": dummy_reward,
            "terminated": is_terminal,
            "truncated": is_truncated,
            # Optional: Add any other info you need
            "info": {"step": step_idx}
        }
        episode.append(timestep_data)
    raw_episodes.append(episode)
print(f"Generated {len(raw_episodes)} dummy episodes.")

# --- 2. Process Raw Data into a Flat List of Timesteps ---
# The `datasets` library generally works best with a flat list or dictionary
# where each element/row represents a single data point (here, a timestep).
# We also need to add the episode index.

processed_data = []
print("Processing raw data into flat list...")
for ep_idx, episode in enumerate(tqdm(raw_episodes, desc="Processing Episodes")):
    for timestep_data in episode:
        # Add episode index to each step
        timestep_data["episode_index"] = np.int64(ep_idx) # Use standard types
        processed_data.append(timestep_data)

print(f"Total timesteps processed: {len(processed_data)}")

# --- 3. Define the Dataset Schema (`datasets.Features`) ---
# This is crucial for defining the structure and types of your dataset.
# Match this to expected LeRobot conventions or your specific needs.
# Using datasets.Image() allows automatic handling of image data (PIL, numpy)

feature_dict = {
    "episode_index": datasets.Value("int64"),
    "observation": {
        "image": datasets.Image(), # Handles PIL Images, NumPy arrays
        "state": datasets.Sequence(datasets.Value("float32"), length=state_dim), # Or length=-1 if variable
    },
    "action": datasets.Sequence(datasets.Value("float32"), length=action_dim),
    "reward": datasets.Value("float32"),
    "terminated": datasets.Value("bool"),
    "truncated": datasets.Value("bool"),
    "info": { # Example nested info dict - define its structure too
        "step": datasets.Value("int64"),
    }
    # Add other features as needed
}

# If you DON'T have images, adjust the observation dict accordingly, e.g.:
# feature_dict = {
#     ...
#     "observation": {
#         "state": datasets.Sequence(datasets.Value("float32"), length=state_dim),
#     },
#     ...
# }


dataset_features = datasets.Features(feature_dict)
print("\nDefined Dataset Features:")
print(dataset_features)

# --- 4. Create the `Dataset` Object ---
# Use from_list for our processed data structure
print("\nCreating Hugging Face Dataset...")
hf_dataset = datasets.Dataset.from_list(processed_data, features=dataset_features)
print("Dataset created successfully!")
print(hf_dataset)

# --- 5. (Optional) Create a DatasetDict for Splits ---
# If you have pre-defined train/validation/test splits of your `processed_data` list
# (e.g., `train_data`, `val_data`), create a DatasetDict:

# Example: Let's just use the same data for simplicity here
# In reality, you'd split your `processed_data` list appropriately
train_dataset = hf_dataset # Replace with your actual train split Dataset object
# val_dataset = datasets.Dataset.from_list(val_data, features=dataset_features)

dataset_dict = datasets.DatasetDict({
    "train": train_dataset,
    # "validation": val_dataset # Add other splits if you have them
})
print("\nCreated DatasetDict:")
print(dataset_dict)

# --- 6. Save Locally or Push to Hub ---

# Option A: Save to disk
output_dir = "./my_new_lerobot_dataset"
print(f"\nSaving dataset to disk at: {output_dir}")
dataset_dict.save_to_disk(output_dir)
print("Dataset saved locally.")

# --- How to load it back ---
# loaded_dataset_dict = datasets.load_from_disk(output_dir)
# print("\nLoaded dataset from disk:")
# print(loaded_dataset_dict)

# Option B: Push to Hugging Face Hub
# Make sure you ran `huggingface-cli login` in your terminal first!
# Choose a unique name for your dataset on the Hub
hub_dataset_name = "your-hf-username/my-new-lerobot-dataset-test" # CHANGE THIS!
print(f"\nAttempting to push dataset to Hub: {hub_dataset_name}")
try:
    # You can push individual datasets or the whole dict
    # Pushing the dict preserves the splits (train, validation, etc.)
    dataset_dict.push_to_hub(hub_dataset_name)
    print(f"Dataset successfully pushed to Hub: https://huggingface.co/datasets/{hub_dataset_name}")
except Exception as e:
    print(f"Failed to push to Hub: {e}")
    print("Ensure you are logged in (`huggingface-cli login`) and the repository name is available.")

# --- Important Considerations ---

# 1.  **Schema Consistency:** Ensure your `datasets.Features` definition accurately reflects your data types and shapes. Consistency is key for LeRobot. Check existing LeRobot datasets on the Hub for common schemas if you want to maximize compatibility.
# 2.  **Data Types:** Use standard types like `np.float32`, `np.int64`, `bool`. The `datasets` library handles conversion where possible, but being explicit is good.
# 3.  **Large Datasets:** For very large datasets that don't fit in RAM, `datasets.Dataset.from_generator` is a more memory-efficient way to create the dataset. You'd write a Python generator function that yields dictionaries for each timestep.
# 4.  **Metadata:** You can add descriptions and other metadata to your dataset:
#    `hf_dataset.info.description = "My awesome new robot dataset description."`
#    `hf_dataset.info.homepage = "http://my-project-page.com"`
#    `hf_dataset.info.license = "mit"` # Use SPDX license identifiers if possible
#    This info is stored with the dataset and displayed on the Hub.
# 5.  **LeRobot Utilities:** The `lerobot` library itself might offer higher-level functions or validation tools specifically for creating datasets intended for the official LeRobot collection or use with its specific dataloaders. Check their documentation if contributing or using their specific tools.