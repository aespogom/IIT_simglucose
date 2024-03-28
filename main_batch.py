import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# List of tasks. Each task is a list where the first element is the script, and the following elements are the arguments for that script.
tasks = [
    # MLP Parallel
    ['python','-m', 'main_MLP', '--pred_horizon 30', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['python','-m', 'main_MLP', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['python','-m', 'main_MLP', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['python','-m', 'main_MLP', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_parallel.nm'],
    ['python','-m', 'main_MLP', '--pred_horizon','30'],
    ['python','-m', 'main_MLP', '--pred_horizon','45'],
    ['python','-m', 'main_MLP', '--pred_horizon','60'],
    ['python','-m', 'main_MLP', '--pred_horizon','120'],
    # MLP Tree
    ['python','-m', 'main_MLP_tree', '--pred_horizon 30', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', "--dump_path","results/MLP_joint"],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', "--dump_path","results/MLP_joint"],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', "--dump_path","results/MLP_joint"],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', "--dump_path","results/MLP_joint"],
    # MLP Tree joint
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon 30', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree_joint_double_cyclic.nm', "--dump_path","results/MLP_joint/double_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree_joint_remove_cyclic.nm', "--dump_path","results/MLP_joint/remove_cyclic"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','30', "--dump_path","results/MLP_joint"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','45', "--dump_path","results/MLP_joint"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','60', "--dump_path","results/MLP_joint"],
    ['python','-m', 'main_MLP_tree_joint', '--pred_horizon','120', "--dump_path","results/MLP_joint"],
    # MLP Tree depth
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm', "--dump_path","results/MLP_tree/module_depth"],
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','30', "--dump_path","results/MLP_tree/module_depth"],
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','45', "--dump_path","results/MLP_tree/module_depth"],
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','60', "--dump_path","results/MLP_tree/module_depth"],
    ['python','-m', 'main_MLP_tree_depth', '--pred_horizon','120', "--dump_path","results/MLP_tree/module_depth"],
    # MLP Scaled
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '30', '--neuro_mapping', 'train_config/MLP_scaled_30.nm'],
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '45', '--neuro_mapping', 'train_config/MLP_scaled_45.nm'],
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '60', '--neuro_mapping', 'train_config/MLP_scaled_60.nm'],
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '120', '--neuro_mapping', 'train_config/MLP_scaled_120.nm'],
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '30'],
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '45'],
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '60'],
    ['python','-m', 'main_MLP_scaled', '--pred_horizon', '120'],
    # LSTM parallel
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '30', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '45', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '60', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '120', '--neuro_mapping', 'train_config/LSTM_parallel.nm'],
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '30'],
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '45'],
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '60'],
    ['python','-m', 'main_LSTM_parallel', '--pred_horizon', '120'],
    # LSTM tree
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '30', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '45', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '60', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '120', '--neuro_mapping', 'train_config/LSTM_tree.nm'],
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '30'],
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '45'],
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '60'],
    ['python','-m', 'main_LSTM_tree', '--pred_horizon', '120'],
    # RNN parallel
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '30', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '45', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '60', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '120', '--neuro_mapping', 'train_config/RNN_parallel.nm'],
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '30'],
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '45'],
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '60'],
    ['python','-m', 'main_RNN_parallel', '--pred_horizon', '120'],
    # RNN tree
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '30', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '45', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '60', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '120', '--neuro_mapping', 'train_config/RNN_tree.nm'],
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '30'],
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '45'],
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '60'],
    ['python','-m', 'main_RNN_tree', '--pred_horizon', '120'],
]

# Function to run a single task
def run_task(task):
    """Run a single task using subprocess.run"""
    print(f"Starting task {task} at {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
    result = subprocess.run(task, shell=True, capture_output=True, text=True)
    print(f"Finish task {task} at {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
    if result.returncode == 0:
        return f"Task {task} completed successfully. Output:\n{result.stdout}"
    else:
        return f"Task {task} failed with error: {result.stderr}"

if __name__ == '__main__':

    # Number of tasks to run simultaneously
    num_workers = 4

    # Using ProcessPoolExecutor to run tasks concurrently
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"{task} generated an exception: {exc}")