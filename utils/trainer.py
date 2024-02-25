import json
import random
from typing import Union
import torch
from torch import nn
from torch.optim import AdamW
import math
import os
import time
from pickle import dump
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('error')  # "error", "ignore", "always", "default", "module" or "once"

from dataset.glucosedataset import GlucoseDataset
from .counterfactual_utils import deserialize_variable_name, parse_variable_name, get_activation_at, logger
from .early_stopper import EarlyStopper

class Trainer:
    def __init__(
        self,
        params: dict,
        dataset: GlucoseDataset, 
        val_dataset: GlucoseDataset,
        neuro_mapping: Union[str,None],
        student: nn.Module,
        teacher: nn.Module
    ):
        logger.info("Initializing Trainer")
        self.params = params
        self.dump_path = params.dump_path
        self.neuro_mapping = neuro_mapping
        self.student = student
        self.teacher = teacher

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0

        # if not neuro_mapping, alpha are not used and regular loss is used as default
        self.alpha_ce = params.alpha_ce
        self.alpha_causal = params.alpha_causal

        self.track_loss = []
        
        if self.neuro_mapping:
            self.last_loss_causal_ce = 0
            self.last_teacher_interchange_efficacy = 0
            self.last_student_interchange_efficacy = 0
            self.track_II_loss = []

            # Deserialize causal neuron mappings
            # $L:X$H:Y$[Z:Z+1]
            # X --> layer number, Y --> head, Z --> location
            # In our case, we dont have encoder so head is "useless"
            self.deserialized_interchange_variable_mappings = []
            with open(self.neuro_mapping) as json_file:
                neuron_mapping = json.load(json_file)
                logger.info(f"Neuron Mapping: {neuron_mapping}")
                interchange_variable_mappings = neuron_mapping["interchange_variable_mappings"]
                for m in interchange_variable_mappings:
                    teacher_deserialized_variables = []
                    for variable in m["teacher_variable_names"]:
                        teacher_deserialized_variables.append(deserialize_variable_name(variable))
                    student_deserialized_variables = []
                    for variable in m["student_variable_names"]:
                        student_deserialized_variables.append(deserialize_variable_name(variable))
                    self.deserialized_interchange_variable_mappings += [
                        [teacher_deserialized_variables, student_deserialized_variables]
                    ]
            logger.info(f"Deserialized interchange variable mappings {str(self.deserialized_interchange_variable_mappings)}.")
        else:
            logger.info('ATTENTION: TRAINING REGULAR, NOT IIT APPLIED!')
        
        logger.info("--- Dataset loaded")
        self.dataloader = dataset
        self.val_dataloader = val_dataset

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 1.0e-5,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=0.01, eps=1e-06, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * 0.05)
        logger.info("------ Warm steps ----- %i" % warmup_steps)
        logger.info("------ num_train_optimization_steps  ----- %i" % num_train_optimization_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )
        self.early_stopper = EarlyStopper(patience=params.patience)
        # self.loss = RMSELoss()
        self.loss = nn.MSELoss(reduction='mean')

    def train(self):
        """
        The real training loop.
        """
        logger.info("Starting training")
        self.last_log = time.time()
        self.teacher.eval()
        self.student.train()
        self.optimizer.zero_grad()

        for _ in range(self.params.n_epoch):
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            iter_bar = tqdm(self.dataloader, desc="-Iter")
            for batch in iter_bar:
                pre_meal, post_meal, pat_names = batch

                source = pre_meal[0, :]
                source_labels = post_meal[0]
                
                base = pre_meal[-1, :]
                base_labels = post_meal[-1]
                
                look_up_source, look_up_base = pat_names

                if self.neuro_mapping:
                    self.step(
                        source_ids=source,
                        source_labels=source_labels,
                        base_ids=base,
                        base_labels=base_labels,
                        look_up_source=look_up_source,
                        look_up_base=look_up_base
                    )
                else:
                    self.regular_step(
                        source_ids=source,
                        source_labels=source_labels,
                        base_ids=base,
                        base_labels=base_labels,
                        look_up_source=look_up_source,
                        look_up_base=look_up_base   
                    )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        # "Last_loss": f"{self.last_loss:.2f}", 
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}",
                        # "Last_ii_acc": f'{self.last_ii_acc:.2f}',
                        # "Avg_cum_ii_acc": f"{self.total_ii_acc_epoch/self.n_iter:.2f}",
                        # "Last_beh_acc": f'{self.last_beh_acc:.2f}',
                        # "Avg_cum_beh_acc": f"{self.total_beh_acc_epoch/self.n_iter:.2f}",
                        "Last_t_efficacy": f"{self.last_teacher_interchange_efficacy:.2f}" if self.neuro_mapping else "--",
                        "Last_s_efficacy": f"{self.last_student_interchange_efficacy:.2f}" if self.neuro_mapping else "--"
                    }
                )
            iter_bar.close()
            logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.early_stopper.early_stop(self.total_loss_epoch/self.n_iter):
                self.end_epoch()          
                break
            self.end_epoch()

        # Save the training loss values
        with open(os.path.join(self.dump_path,'train_loss.pkl'), 'wb') as file:
            dump(self.track_loss, file)
        
        # Save the II loss values
        if self.neuro_mapping:
            with open(os.path.join(self.dump_path,'ii_loss.pkl'), 'wb') as file:
                dump(self.track_II_loss, file)

        logger.info("Training is finished")

    def step(
        self,
        source_ids: torch.tensor, 
        source_labels: torch.tensor,
        look_up_source,
        base_ids: torch.tensor,
        base_labels: torch.tensor,
        look_up_base
    ):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).
        Input:
        ------
       """
        # we randomly select the pool of neurons to interchange.
        selector = random.randint(0, len(self.deserialized_interchange_variable_mappings)-1)
        interchange_variable_mapping = self.deserialized_interchange_variable_mappings[selector]
        teacher_variable_names = random.choice(interchange_variable_mapping[0])
        student_variable_names = random.choice(interchange_variable_mapping[1])

        teacher_interchanged_variables_mapping = {}
        student_interchanged_variables_mapping = {}
        # we store the interchange here.
        for i, variable in enumerate(teacher_variable_names):
            layer_index, LOC = parse_variable_name(variable)
            if layer_index in teacher_interchanged_variables_mapping:
                teacher_interchanged_variables_mapping[layer_index] += [(i, LOC)]
            else:
                teacher_interchanged_variables_mapping[layer_index] = [(i, LOC)]
        for i, variable in enumerate(student_variable_names):
            layer_index, LOC = parse_variable_name(variable)
            if layer_index in student_interchanged_variables_mapping:
                student_interchanged_variables_mapping[layer_index] += [(i, LOC)]
            else:
                student_interchanged_variables_mapping[layer_index] = [(i, LOC)]
        
        with torch.no_grad():
            # teacher forward pass normal.
            teacher_outputs = self.teacher(
                input_ids=source_ids, # source input
                look_up = look_up_source
            )
            # teacher forward pass normal
            dual_teacher_outputs = self.teacher(
                input_ids=base_ids, # base input
                look_up=look_up_base
            )
            # teacher forward pass for interchange variables.
            dual_counterfactual_activations_teacher = get_activation_at(
                self.teacher,
                input_ids=base_ids, # this is different! OBTAIN BASE ACTIVATIONS
                variable_names=teacher_variable_names,
                look_up = look_up_base
            )

            # teacher forward pass for interchanged outputs.
            counterfactual_outputs_teacher = self.teacher(
                input_ids=source_ids, # source inputs 
                look_up = look_up_source,
                interchanged_variables=dual_counterfactual_activations_teacher, # base activations
                variable_names=teacher_interchanged_variables_mapping
            )
            #
            counterfactual_activations_teacher = get_activation_at(
                self.teacher,
                input_ids=source_ids, # this is different! OBTAIN SOURCE ACTIVATIONS
                variable_names=teacher_variable_names,
                look_up = look_up_source
            )
            #
            dual_counterfactual_outputs_teacher = self.teacher(
                input_ids=base_ids, # base inputs 
                look_up = look_up_base,
                interchanged_variables=counterfactual_activations_teacher, # source activations
                variable_names=teacher_interchanged_variables_mapping
            )

        t_outputs = teacher_outputs["outputs"]
        dual_t_outputs = dual_teacher_outputs["outputs"]
        # student forward pass normal.
        student_outputs = self.student(
            input_ids=source_ids, # source input
            t_outputs=t_outputs,
            #lm_labels
            #t_hidden
        )
        # student forward pass normal.
        dual_student_outputs = self.student(
            input_ids=base_ids, # base input
            t_outputs=dual_t_outputs
            #lm_labels
            #dual_t_hidden
        )

        s_outputs = student_outputs["outputs"]
        dual_s_outputs = dual_student_outputs["outputs"]
        causal_t_outputs = counterfactual_outputs_teacher["outputs"]
        ## HERE ANA IS THIS A BUG IN ORIGINAL CODE??? they are using counterfactual_outputs_teacher instead of dual_counterfactual_outputs_teacher
        dual_causal_t_outputs = dual_counterfactual_outputs_teacher["outputs"]
        
        # Loss_ce
        loss_ce = student_outputs["loss"]
        loss_ce += dual_student_outputs["loss"]
        loss = self.alpha_ce * loss_ce

        self.track_loss.append(loss.item())

        # student forward pass for interchange variables.
        dual_counterfactual_activations_student = get_activation_at(
            self.student,
            base_ids, # this is different! OBTAIN BASE ACTIVATIONS
            variable_names=student_variable_names
        )
        counterfactual_activations_student = get_activation_at(
            self.student,
            source_ids, # this is different! OBTAIN SOURCE ACTIVATIONS
            variable_names=student_variable_names
        )

        # student forward pass for interchanged outputs.
        counterfactual_outputs_student = self.student(
            input_ids=source_ids, # source input
            # intervention
            interchanged_variables=dual_counterfactual_activations_student, # base activations
            variable_names=student_interchanged_variables_mapping,
            # backward loss.
            t_outputs=t_outputs,
            #t_hidden
            causal_t_outputs=causal_t_outputs,
            #causal_t_hidden
            s_outputs=s_outputs
            #s_hidden
        )
        dual_counterfactual_outputs_student = self.student(
            input_ids=base_ids, # base input
            # intervention
            interchanged_variables=counterfactual_activations_student, # source activations
            variable_names=student_interchanged_variables_mapping,
            # backward loss.
            t_outputs=dual_t_outputs,
            #dual_t_hidden
            causal_t_outputs=dual_causal_t_outputs,
            #dual_causal_t_hidden
            s_outputs=dual_s_outputs
            #dual_causal_s_hidden
        )
        causal_loss_ce = counterfactual_outputs_student["loss"]
        causal_loss_ce += dual_counterfactual_outputs_student["loss"]

        self.last_student_interchange_efficacy = counterfactual_outputs_student["student_interchange_efficacy"].item()
        self.last_teacher_interchange_efficacy = counterfactual_outputs_student["teacher_interchange_efficacy"].item()
        
        self.last_student_interchange_efficacy += dual_counterfactual_outputs_student["student_interchange_efficacy"].item()
        self.last_teacher_interchange_efficacy += dual_counterfactual_outputs_student["teacher_interchange_efficacy"].item()
        
        loss += self.alpha_causal * causal_loss_ce
            
        self.track_II_loss.append(counterfactual_outputs_student["loss"].item()+dual_counterfactual_outputs_student["loss"].item() )
        
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()# optional recording of the value.
        self.last_loss_causal_ce = causal_loss_ce.item()

        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            print(f"Student outputs is: {str(s_outputs)}")
            print(f"Teacher outputs is: {str(t_outputs)}")
            print(f"Student dual outputs is: {str(dual_s_outputs)}")
            print(f"Teacher dual outputs is: {str(dual_t_outputs)}")
        
        self.optimize(loss)


    def regular_step(
        self,
        source_ids: torch.tensor, 
        source_labels: torch.tensor,
        look_up_source,
        base_ids: torch.tensor,
        base_labels: torch.tensor,
        look_up_base
    ):
        """
        One optimization step: forward of student, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).
        Input:
        ------
       """
        with torch.no_grad():
            # teacher forward pass normal.
            teacher_outputs = self.teacher(
                input_ids=source_ids, # source input
                look_up = look_up_source
            )
            # teacher forward pass normal
            dual_teacher_outputs = self.teacher(
                input_ids=base_ids, # base input
                look_up=look_up_base
            )

        t_outputs = teacher_outputs["outputs"]
        dual_t_outputs = dual_teacher_outputs["outputs"]
        # student forward pass normal.
        student_outputs = self.student(
            input_ids=source_ids, # source input
            t_outputs=t_outputs,
            #lm_labels
            #t_hidden
        )
        # student forward pass normal.
        dual_student_outputs = self.student(
            input_ids=base_ids, # base input
            t_outputs=dual_t_outputs
            #lm_labels
            #dual_t_hidden
        )

        s_outputs = student_outputs["outputs"]
        dual_s_outputs = dual_student_outputs["outputs"]
        
        # Loss
        loss = student_outputs["loss"]
        loss+= dual_student_outputs["loss"]

        self.track_loss.append(loss.item())
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()# optional recording of the value.

        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            print(f"Student outputs is: {str(s_outputs)}")
            print(f"Student dual outputs is: {str(dual_s_outputs)}")
        
        self.optimize(loss)


    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        loss.backward()

        #iter()
        self.n_iter += 1
        self.n_total_iter += 1
        self.last_log = time.time()

        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
    
    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        #f"model_epoch_{self.epoch}_loss_{self.total_loss_epoch/self.n_iter:.2f}_ii_acc_{self.total_ii_acc_epoch/self.n_iter:.2f}_beh_acc_{self.total_beh_acc_epoch/self.n_iter:.2f}
        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}_loss_{self.total_loss_epoch/self.n_iter:.2f}.pth")

        self.epoch += 1
        self.n_iter = 0
        self.total_loss_epoch = 0
        self.total_ii_acc_epoch = 0
        self.total_beh_acc_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        current_checkpoints = os.listdir(self.dump_path)
        current_loss_checkpoint = [float(loss.split('_')[-1].split('.pth')[0]) for loss in current_checkpoints if "model_epoch" in loss]
        if all([loss_checkpoint >= self.total_loss_epoch/self.n_iter for loss_checkpoint in current_loss_checkpoint]):
            if len(current_checkpoints)>0:
                [os.remove(os.path.join(self.dump_path, file)) for file in current_checkpoints if "model_epoch" in file]
        mdl_to_save = self.student
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
    
    def evaluate(self):
        self.student.eval()
        if self.neuro_mapping:
            self.teacher.eval()
        current_checkpoints = os.listdir(self.dump_path)
        current_loss_checkpoint = [float(loss.split('_')[-1].split('.pth')[0]) for loss in current_checkpoints if "model_epoch" in loss]
        min_loss = min(current_loss_checkpoint)
        best_checkpoint = [checkpoint for checkpoint in current_checkpoints if str(min_loss) in checkpoint][0]
        self.student.load_state_dict(torch.load(os.path.join(self.dump_path,best_checkpoint)))
        ## TODO EVALUATE METHODS
        ii_loss = self.ii_loss() if self.neuro_mapping else None
        beh_loss = self.beh_loss()
        logger.info(f"------------ II LOSS {ii_loss}") if self.neuro_mapping else None

        logger.info(f"------------- BEH LOSS {beh_loss}")

    def ii_loss(self):
        """ Interchange intervention loss quantifies the extent to which the interpretable
        causal model is a proxy for the network"""

        
        labels = []
        predictions = []
        with torch.no_grad():
            for batch in self.val_dataloader:

                # we randomly select the pool of neurons to interchange.
                selector = random.randint(0, len(self.deserialized_interchange_variable_mappings)-1)
                interchange_variable_mapping = self.deserialized_interchange_variable_mappings[selector]
                teacher_variable_names = random.choice(interchange_variable_mapping[0])
                student_variable_names = random.choice(interchange_variable_mapping[1])

                teacher_interchanged_variables_mapping = {}
                student_interchanged_variables_mapping = {}
                # we store the interchange here.
                for i, variable in enumerate(teacher_variable_names):
                    layer_index, LOC = parse_variable_name(variable)
                    if layer_index in teacher_interchanged_variables_mapping:
                        teacher_interchanged_variables_mapping[layer_index] += [(i, LOC)]
                    else:
                        teacher_interchanged_variables_mapping[layer_index] = [(i, LOC)]
                for i, variable in enumerate(student_variable_names):
                    layer_index, LOC = parse_variable_name(variable)
                    if layer_index in student_interchanged_variables_mapping:
                        student_interchanged_variables_mapping[layer_index] += [(i, LOC)]
                    else:
                        student_interchanged_variables_mapping[layer_index] = [(i, LOC)]

                pre_meal, post_meal, pat_names = batch

                source = pre_meal[0, :]
                source_labels = post_meal[0]
                
                base = pre_meal[-1, :]
                base_labels = post_meal[-1]
                
                look_up_source, look_up_base = pat_names
            
                # Run the causal model with the intervention:
                dual_counterfactual_activations_teacher = get_activation_at(
                    self.teacher,
                    input_ids=base, # this is different! OBTAIN BASE ACTIVATIONS
                    labels=base_labels,
                    variable_names=teacher_variable_names,
                    look_up=look_up_base
                )
                outputs_teacher = self.teacher(
                    input_ids=source, # source input
                    look_up=look_up_source,
                    labels=source_labels,
                    # intervention
                    interchanged_variables=dual_counterfactual_activations_teacher, # base activations
                    variable_names=teacher_interchanged_variables_mapping
                )
                labels.append(outputs_teacher["outputs"])

                # Run the neural model with the intervention:
                dual_counterfactual_activations_student = get_activation_at(
                    self.student,
                    base, # this is different! OBTAIN BASE ACTIVATIONS
                    variable_names=student_variable_names
                )
                outputs_student = self.student(
                    input_ids=source, # source input
                    # intervention
                    interchanged_variables=dual_counterfactual_activations_student, # base activations
                    variable_names=student_interchanged_variables_mapping
                )
                # Get the neural model's prediction with the intervention:
                pred = outputs_student['outputs']
                predictions.append(pred)

            logger.info("Counterfactual evaluation")
            logger.info(torch.cat(predictions, dim=0))
            logger.info("Labels")
            logger.info(torch.stack(labels))
            return self.loss(torch.cat(predictions, dim=0),torch.stack(labels))
    
    def beh_loss(self):
        """ Behavioral loss is the percentage of inputs that student agrees with teacher """
        labels = []
        predictions = []
        with torch.no_grad():
            for batch in self.val_dataloader:

                pre_meal, post_meal, pat_names = batch

                source = pre_meal[0, :]
                source_labels = post_meal[0]
                
                base = pre_meal[-1, :]
                base_labels = post_meal[-1]
                
                look_up_source, look_up_base = pat_names

                # Run the causal model:
                outputs_teacher = self.teacher(
                    input_ids=source, # source input
                    labels=source_labels,
                    look_up=look_up_source
                )
                labels.append(outputs_teacher["outputs"])
                # Run the neural model:
                outputs_student = self.student(
                    input_ids=source # source input
                )
                # Get the neural model's prediction
                pred = outputs_student['outputs']
                predictions.append(pred)

                # Run the causal model:
                outputs_teacher = self.teacher(
                    input_ids=base, # base input
                    labels=base_labels,
                    look_up=look_up_base
                )
                labels.append(outputs_teacher["outputs"])
                # Run the neural model:
                outputs_student = self.student(
                    input_ids=base # base input
                )
                # Get the neural model's prediction
                pred = outputs_student['outputs']
                predictions.append(pred)

            logger.info("\nStandard evaluation")
            logger.info("Predictions")
            logger.info(torch.cat(predictions, dim=0))
            logger.info("Labels")
            logger.info(torch.stack(labels))
            return self.loss(torch.cat(predictions, dim=0),torch.stack(labels))
        
