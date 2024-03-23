import subprocess

# List of tasks. Each task is a list where the first element is the script, and the following elements are the arguments for that script.
tasks = [
    # ['main_MLP_tree_joint.py', '--pred_horizon 30', '--neuro_mapping train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon 45', '--neuro_mapping train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon 60', '--neuro_mapping train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon 120', '--neuro_mapping train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon 30', '--neuro_mapping train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path results/MLP_joint/remove_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon 45', '--neuro_mapping train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path results/MLP_joint/remove_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon 60', '--neuro_mapping train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path results/MLP_joint/remove_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon 120', '--neuro_mapping train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path results/MLP_joint/remove_cyclic"]
    ['main_MLP_tree_joint.py', '--pred_horizon 30', "--dump_path results/MLP_joint"],
    ['main_MLP_tree_joint.py', '--pred_horizon 45', "--dump_path results/MLP_joint"],
    ['main_MLP_tree_joint.py', '--pred_horizon 60', "--dump_path results/MLP_joint"],
    ['main_MLP_tree_joint.py', '--pred_horizon 120', "--dump_path results/MLP_joint"],
   # Add more tasks as needed
]

for task in tasks:
    # Use subprocess.run to execute the script with arguments.
    # *task unpacks the task list into arguments for subprocess.run.
    result = subprocess.run(['python'] + task, capture_output=True, text=True)

    # Check if the script executed successfully
    if result.returncode != 0:
        print(f"Task {task} failed with error: {result.stderr}")
        # Uncomment the next line if you want to stop execution on failure
        # break
    else:
        print(f"Task {task} completed successfully. Output:\n{result.stdout}")

# You can customize the tasks list and error handling as per your requirements.
