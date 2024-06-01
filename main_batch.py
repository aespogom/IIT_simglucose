import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# List of tasks. Each task is a list where the first element is the script, and the following elements are the arguments for that script.
tasks = [
    # MLP Tree
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '101', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--seed', '101', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '101', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--seed', '101', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--seed', '101', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--seed', '101', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '101', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '101', '--date_experiment', '2024-05-30'],

    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '33', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '33', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--seed', '33', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--seed', '33', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--seed', '33', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--seed', '33', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '33', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '33', '--date_experiment', '2024-05-30'],

    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '78', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '78', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--seed', '78', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--seed', '78', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--seed', '78', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--seed', '78', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '78', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '78', '--date_experiment', '2024-05-30'],

    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '42', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '42', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--seed', '42', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--seed', '42', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--seed', '42', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '42', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '42', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--seed', '42', '--date_experiment', '2024-05-30'],

    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '89', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '89', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--seed', '89', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--seed', '89', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--seed', '89', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--seed', '89', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '89', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '89', '--date_experiment', '2024-05-30'],

    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '61', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '61', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','45', '--seed', '61', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','60', '--seed', '61', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--seed', '61', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '61', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','30', '--neuro_mapping','train_config/MLP_tree.nm','--seed', '61', '--date_experiment', '2024-05-30'],
    ['python','-m', 'main_MLP_tree', '--pred_horizon','120', '--seed', '61', '--date_experiment', '2024-05-30'],

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