import argparse
from datetime import datetime
import json
import os
import shutil
import torch
from dataset.glucosedataset import setup_loaders
from models.student import MLP as student
from models.teacher import simglucose as teacher
from utils.counterfactual_utils import set_seed, logger
from utils.trainer import Trainer

def prepare_trainer(args):

    # ARGS #
    args.seed=56
    set_seed(args)
    
    if os.path.exists(args.dump_path):
        shutil.rmtree(args.dump_path)

    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

    # SAVE PARAMS #
    logger.info(f"Param: {args}")
    with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    student_model = student.MLP()
    # student = student_model.to(f"cuda:0", non_blocking=True)
    logger.info("Student loaded.")
    teacher_model = teacher.Simglucose()
    # teacher = teacher_model.to(f"cuda:0", non_blocking=True)
    logger.info("Teacher loaded.")

    # DATA LOADER
    train_dataset, val_dataset, _ = setup_loaders()
    logger.info("Data loader created.")

    # TRAINER #
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    trainer = Trainer(
        params=args,
        dataset=train_dataset,
        val_dataset=val_dataset,
        neuro_mapping=None,
        student=student_model,
        teacher=teacher_model
    )
    logger.info("trainer initialization done.")
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite dump_path if it already exists."
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        default="results",
        help="The output directory (log, checkpoints, parameters, etc.)"
    )
    parser.add_argument("--n_epoch", type=int, default=800, help="Number of epochs.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=20,
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument(
        "--batch_train",
        type=int,
        default=20,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--batch_val",
        type=int,
        default=20,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for early stopper.",
    )

    
    args = parser.parse_args()
    
    today = datetime.today().strftime('%Y-%m-%d')
    start = datetime.now()
    # config the runname here and overwrite.
    run_name = f"s_MLP_data_insilico_seed_56_{today}"
    args.run_name = run_name
    args.dump_path = os.path.join(args.dump_path, args.run_name)
    trainer = prepare_trainer(args)
    logger.info("Start training.")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Something went wrong :( --> {e}")
    finally:
        ## TODO EVALUATE METHODS
        trainer.evaluate()
    print(f'Finished at {start - datetime.now()}')
