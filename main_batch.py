import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
import torch

# Check available GPUs
NUM_GPUS = torch.cuda.device_count()
print(torch.cuda.is_available())
print(f"Available GPUs: {NUM_GPUS}")

PYTHON_EXECUTABLE = '.venv\\Scripts\\python.exe'
# List of tasks. Each task is a list where the first element is the script, and the following elements are the arguments for that script.
# python -m main_MLP --pred_horizon 30 --neuro_mapping train_config/MLP_parallel.nm --seed 44 --student_model parallel
tasks = []
for seed in [44,87]:
    for model in ["parallel","tree", "tree_joint"]:
        tasks.extend([
            
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '30',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '30',
                '--seed', str(seed),
                '--student_model', model
            ],
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '45',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '45',
                '--seed', str(seed),
                '--student_model', model
            ],
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '60',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '60',
                '--seed', str(seed),
                '--student_model', model
            ],
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '120',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            [PYTHON_EXECUTABLE, '-m', 'main_MLP',
                '--pred_horizon', '120',
                '--seed', str(seed),
                '--student_model', model
            ]
        ])

# Function to run a single task
def run_task(task, gpu_id):
    """Run a single task using subprocess.run"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    start_time = datetime.now()
    print(f"Starting task {task} at {start_time.strftime('%d/%m/%Y, %H:%M:%S')}")

    result = subprocess.run(task, shell=False, capture_output=True, text=True, env=env)
    
    end_time = datetime.now()
    print(f"Finished task {task} at {end_time.strftime('%d/%m/%Y, %H:%M:%S')} (Duration: {end_time - start_time})")
    
    if result.returncode == 0:
        return f"Task {task} completed successfully. Output:\n{result.stdout}"
    else:
        return f"Task {task} failed with error: {result.stderr}"

if __name__ == '__main__':
    num_workers = NUM_GPUS or 2  # Default to 2 workers if no GPU
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            gpu_id = i % max(NUM_GPUS, 1)
            futures[executor.submit(run_task, task, gpu_id)] = task

        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"{task} generated an exception: {exc}")