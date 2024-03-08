import argparse
from datetime import datetime
import json
import os
import shutil
import torch
from dataset.glucosedataset import setup_loaders
from models.teacher import simglucose as teacher
from models.student import MLP_scaled as student
from utils.counterfactual_utils import set_seed, logger
from utils.trainer import Trainer

def prepare_trainer(args):

    # ARGS #
    args.seed=56
    set_seed(args)
    
    # if os.path.exists(args.dump_path):
    #     shutil.rmtree(args.dump_path)

    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

    # SAVE PARAMS #
    logger.info(f"Param: {args}")
    with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # EX: Pred horizon 30 with a time step of 3 minutes represent 10 integrations in the simulator --> 1 initial block + 9 scaled
    student_model = student.MLP_scaled(args.input_size, args.output_size, args.pred_horizon/3)
    # student = student_model.to(f"cuda:0", non_blocking=True)
    logger.info("Student loaded.")
    teacher_model = teacher.Simglucose(args.pred_horizon, True)
    # teacher = teacher_model.to(f"cuda:0", non_blocking=True)
    logger.info("Teacher loaded.")

    # DATA LOADER
    train_dataset, val_dataset, test_dataset = setup_loaders()
    logger.info("Data loader created.")

    # TRAINER #
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    trainer = Trainer(
        params=args,
        dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        neuro_mapping=args.neuro_mapping,
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
        default=os.path.join("results","MLP_scaled"),
        help="The output directory (log, checkpoints, parameters, etc.)"
    )
    parser.add_argument(
        "--neuro_mapping",
        type=str,
        # default=f"train_config/MLP_scaled_45.nm",
        default=None,
        help="Predefined neuron mapping for the interchange experiment.",
    )
    parser.add_argument(
        "--alpha_ce",
        type=float,
        default=0.25,
        help="Coefficient regular loss",
    )
    parser.add_argument(
        "--alpha_causal",
        type=float,
        default=0.75,
        help="Coefficient causal loss",
    )
    parser.add_argument("--n_epoch", type=int, default=300, help="Number of epochs.")
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
        "--input_size",
        type=int,
        default=20,
        help="Input size",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=64,
        help="Hidden size",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for early stopper.",
    )
    parser.add_argument(
        "--pred_horizon",
        type=int,
        choices=[30, 45, 60, 120],
        help="Prediction horizon.",
    )
    parser.add_argument(
        "--date_experiment",
        type=str,
        default=datetime.today().strftime('%Y-%m-%d'),
        help="Date of the experiemnt in format YYYY-MM-DD."
    )
    
    args = parser.parse_args()
    
    # config the runname here and overwrite.
    run_name = f"s_MLP_scaled_t_simglucose_data_insilico_seed_56_{args.date_experiment}_PH_{str(args.pred_horizon)}" if args.neuro_mapping else f"s_MLP_scaled_data_insilico_seed_56_{args.date_experiment}_PH_{str(args.pred_horizon)}"
    args.run_name = run_name
    args.dump_path = os.path.join(args.dump_path, args.run_name)
    trainer = prepare_trainer(args)
    logger.info("Start training.")
    try:
        if args.date_experiment == datetime.today().strftime('%Y-%m-%d'):
            trainer.train()
        else:
            pass
    except Exception as e:
        logger.error(f"Something went wrong :( --> {e}")
    finally:
        ## TODO EVALUATE METHODS
        trainer.evaluate()
        trainer.test()
