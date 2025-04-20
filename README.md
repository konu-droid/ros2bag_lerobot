1. install lerobot 
2. pip install datasets numpy Pillow huggingface_hub
3. 

### Notes:
1. We trigger the step creation process for praquet files only when a image is availbale, in this way the dataset is at the same frequency as the image topic. This is done assuming that image data will be the slowest or the least frequent data avilable
2. If 'STATE_MODALITIES' is empty a sample state of single_arm and gripper will be filled. Where the single_arm will contain all the joint except the last and the last joint will be put in as gripper joint.
3. 2. If 'ACTION_MODALITIES' is empty a sample action of single_arm and gripper will be filled. Where the single_arm will contain all the joint except the last and the last joint will be put in as gripper joint. 

### usage

for moveit2 just run 'ros2 launch so_arm100_moveit_config demo.launch.py' afteer building the ros2 package.
