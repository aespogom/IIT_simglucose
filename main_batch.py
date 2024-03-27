import subprocess

# List of tasks. Each task is a list where the first element is the script, and the following elements are the arguments for that script.
tasks = [
    # Parallel
    ['main_MLP.py', '--pred_horizon 30', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['main_MLP.py', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['main_MLP.py', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['main_MLP.py', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['main_MLP.py', '--pred_horizon','30'],
    ['main_MLP.py', '--pred_horizon','45'],
    ['main_MLP.py', '--pred_horizon','60'],
    ['main_MLP.py', '--pred_horizon','120'],
    # Tree
    ['main_MLP_tree.py', '--pred_horizon 30', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['main_MLP_tree.py', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['main_MLP_tree.py', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['main_MLP_tree.py', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['main_MLP_tree.py', '--pred_horizon','30', "--dump_path","results/MLP_joint"],
    ['main_MLP_tree.py', '--pred_horizon','45', "--dump_path","results/MLP_joint"],
    ['main_MLP_tree.py', '--pred_horizon','60', "--dump_path","results/MLP_joint"],
    ['main_MLP_tree.py', '--pred_horizon','120', "--dump_path","results/MLP_joint"],
    # Tree joint
    ['main_MLP_tree_joint.py', '--pred_horizon 30', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['main_MLP_tree_joint.py', '--pred_horizon','30', "--dump_path","results/MLP_joint"],
    ['main_MLP_tree_joint.py', '--pred_horizon','45', "--dump_path","results/MLP_joint"],
    ['main_MLP_tree_joint.py', '--pred_horizon','60', "--dump_path","results/MLP_joint"],
    ['main_MLP_tree_joint.py', '--pred_horizon','120', "--dump_path","results/MLP_joint"],
    # Tree depth
    ['main_MLP_tree_depth.py', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['main_MLP_tree_depth.py', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['main_MLP_tree_depth.py', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['main_MLP_tree_depth.py', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['main_MLP_tree_depth.py', '--pred_horizon','30', "--dump_path","results/MLP_tree/module_depth"],
    ['main_MLP_tree_depth.py', '--pred_horizon','45', "--dump_path","results/MLP_tree/module_depth"],
    ['main_MLP_tree_depth.py', '--pred_horizon','60', "--dump_path","results/MLP_tree/module_depth"],
    ['main_MLP_tree_depth.py', '--pred_horizon','120', "--dump_path","results/MLP_tree/module_depth"],
    # Scaled
    ['main_MLP_scaled.py', '--pred_horizon', '30', '--neuro_mapping', 'train_config/MLP_scaled_30.nm'],
    ['main_MLP_scaled.py', '--pred_horizon', '45', '--neuro_mapping', 'train_config/MLP_scaled_45.nm'],
    ['main_MLP_scaled.py', '--pred_horizon', '60', '--neuro_mapping', 'train_config/MLP_scaled_60.nm'],
    ['main_MLP_scaled.py', '--pred_horizon', '120', '--neuro_mapping', 'train_config/MLP_scaled_120.nm'],
    ['main_MLP_scaled.py', '--pred_horizon', '30'],
    ['main_MLP_scaled.py', '--pred_horizon', '45'],
    ['main_MLP_scaled.py', '--pred_horizon', '60'],
    ['main_MLP_scaled.py', '--pred_horizon', '120'],
    # LSTM parallel
    ['main_LSTM_parallel.py', '--pred_horizon', '30', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['main_LSTM_parallel.py', '--pred_horizon', '45', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['main_LSTM_parallel.py', '--pred_horizon', '60', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['main_LSTM_parallel.py', '--pred_horizon', '120', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['main_LSTM_parallel.py', '--pred_horizon', '30'],
    ['main_LSTM_parallel.py', '--pred_horizon', '45'],
    ['main_LSTM_parallel.py', '--pred_horizon', '60'],
    ['main_LSTM_parallel.py', '--pred_horizon', '120'],
    # LSTM tree
    ['main_LSTM_tree.py', '--pred_horizon', '30', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['main_LSTM_tree.py', '--pred_horizon', '45', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['main_LSTM_tree.py', '--pred_horizon', '60', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['main_LSTM_tree.py', '--pred_horizon', '120', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['main_LSTM_tree.py', '--pred_horizon', '30'],
    ['main_LSTM_tree.py', '--pred_horizon', '45'],
    ['main_LSTM_tree.py', '--pred_horizon', '60'],
    ['main_LSTM_tree.py', '--pred_horizon', '120'],
    # RNN parallel
    ['main_RNN_parallel.py', '--pred_horizon', '30', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['main_RNN_parallel.py', '--pred_horizon', '45', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['main_RNN_parallel.py', '--pred_horizon', '60', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['main_RNN_parallel.py', '--pred_horizon', '120', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['main_RNN_parallel.py', '--pred_horizon', '30'],
    ['main_RNN_parallel.py', '--pred_horizon', '45'],
    ['main_RNN_parallel.py', '--pred_horizon', '60'],
    ['main_RNN_parallel.py', '--pred_horizon', '120'],
    # RNN tree
    ['main_RNN_tree.py', '--pred_horizon', '30', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['main_RNN_tree.py', '--pred_horizon', '45', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['main_RNN_tree.py', '--pred_horizon', '60', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['main_RNN_tree.py', '--pred_horizon', '120', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['main_RNN_tree.py', '--pred_horizon', '30'],
    ['main_RNN_tree.py', '--pred_horizon', '45'],
    ['main_RNN_tree.py', '--pred_horizon', '60'],
    ['main_RNN_tree.py', '--pred_horizon', '120'],
]

for task in tasks:
    # Use subprocess.run to execute the script with arguments.
    # *task unpacks the task list into arguments for subprocess.run.
    print(f"Task {task} starts running...")
    result = subprocess.run(['python'] + task, capture_output=True, text=True)

    # Check if the script executed successfully
    if result.returncode != 0:
        print(f"Task {task} failed with error: {result.stderr}")
        # Uncomment the next line if you want to stop execution on failure
        # break
    else:
        print(f"Task {task} completed successfully. Output:\n{result.stdout}")

# You can customize the tasks list and error handling as per your requirements.
