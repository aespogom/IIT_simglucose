import subprocess

# List of tasks. Each task is a list where the first element is the script, and the following elements are the arguments for that script.
tasks = [
    ['main_MLP_scaled.py', '--pred_horizon 30', '--neuro_mapping train_config/MLP_scaled_30.nm'],
    ['main_MLP_scaled.py', '--pred_horizon 45', '--neuro_mapping train_config/MLP_scaled_45.nm'],
    ['main_MLP_scaled.py', '--pred_horizon 60', '--neuro_mapping train_config/MLP_scaled_60.nm'],
    ['main_MLP_scaled.py', '--pred_horizon 120', '--neuro_mapping train_config/MLP_scaled_120.nm'],
    ['main_MLP_scaled.py', '--pred_horizon 30'],
    ['main_MLP_scaled.py', '--pred_horizon 45'],
    ['main_MLP_scaled.py', '--pred_horizon 60'],
    ['main_MLP_scaled.py', '--pred_horizon 120'],
   # Add more tasks as needed
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
