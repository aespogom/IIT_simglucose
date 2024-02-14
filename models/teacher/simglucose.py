## TEACHER MODEL
from datetime import datetime, timedelta
import torch
from torch import nn
import numpy as np
from utils.t1dpatient import T1DPatient
from utils.env import T1DSimEnv
from utils.controller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario


class Simglucose(nn.Module):
    """
    """
    def __init__(self, pred_horizon):
        super().__init__()
        # Oracle simulator
        self.model = self.simulator
        self.loss = nn.MSELoss(reduction='mean')
        # specify start_time as the beginning of today
        now = datetime.now()
        self.start_time = datetime.combine(now.date(), datetime.min.time())
        self.pred_horizon = pred_horizon

    def simulator(self, 
                  pat_name, 
                  meal_size,
                  bolus,
                  # for interchange.
                  interchanged_variables=None,
                  variable_names=None,
                  interchanged_activations=None):
        # Create a simulation environment
        patient = T1DPatient.withName(name=pat_name)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        # custom scenario is a list of tuples (time in hours, meal_size)
        scen = [(0.5, meal_size)]
        scenario = CustomScenario(start_time=self.start_time, scenario=scen)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()

        controller.reset()
        obs, reward, done, info = env.reset()

        if self.pred_horizon == 30:
            sim_time = timedelta(hours=1)
        elif self.pred_horizon == 45:
            sim_time = timedelta(hours=1, minutes=15)
        elif self.pred_horizon == 60:
            sim_time = timedelta(hours=1, minutes=30)
        else:
            sim_time=0

        while env.time < scenario.start_time + sim_time:
            action = controller.policy(obs, reward, done, **info)
            obs, reward, done, info = env.step(action=action,
                                               interchanged_variables=interchanged_variables,
                                               variable_names=variable_names,
                                               interchanged_activations=interchanged_activations)
            info['bolus']=bolus
        return np.array(patient.state_hist)


    def forward(
        self,
        input_ids,
        labels=None,
        look_up = None,
        # for interchange.
        interchanged_variables=None, 
        variable_names=None,
        interchanged_activations=None
    ):
        """
        Inputs:
            input_ids: pre meal parameters
            labels: post meal parameters
            look_up: patient name
            interchanged_variables: alignment,
            variable_names: mapping
            interchanged_activations: values to interchange
        """

        teacher_ouputs = {}
        teacher_ouputs["hidden_states"]=[]
        # we perform the interchange intervention
        meal_size = float(input_ids[-11])
        bolus = float(input_ids[-12])
        x = self.simulator(look_up,
                           meal_size,
                           bolus,
                           variable_names=variable_names,
                           interchanged_variables=interchanged_variables,
                           interchanged_activations=interchanged_activations)

        teacher_ouputs["hidden_states"] = np.transpose(x)[:,::3]

        if self.pred_horizon == 30:
            output_size = -10
        elif self.pred_horizon == 45:
            output_size = -15
        elif self.pred_horizon == 60:
            output_size = -20
        else:
            output_size=0
        
        teacher_ouputs["outputs"]=torch.tensor(teacher_ouputs["hidden_states"][-1,output_size:]*0.01, dtype=torch.float32)

        return teacher_ouputs
