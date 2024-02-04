## TEACHER MODEL
from datetime import datetime, timedelta
import torch
from torch import nn
import numpy as np
from utils.t1dpatient import T1DPatient
from utils.env import T1DSimEnv
from utils.controller import BBController
from utils.loss_fn import RMSELoss
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario


class Simglucose(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        # Oracle simulator
        self.model = self.simulator
        self.loss = RMSELoss()
        # specify start_time as the beginning of today
        now = datetime.now()
        self.start_time = datetime.combine(now.date(), datetime.min.time())

    def simulator(self, 
                  pat_name, 
                  meal_size,
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
        while env.time < scenario.start_time + timedelta(hours=1):
            action = controller.policy(obs, reward, done, **info)
            obs, reward, done, info = env.step(action=action,
                                               interchanged_variables=interchanged_variables,
                                               variable_names=variable_names,
                                               interchanged_activations=interchanged_activations)
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
        meal_size = float(input_ids[-1,2])
        x = self.simulator(look_up,
                           meal_size,
                           variable_names=variable_names,
                           interchanged_variables=interchanged_variables,
                           interchanged_activations=interchanged_activations)

        teacher_ouputs["hidden_states"] = x
        
        teacher_ouputs["outputs"]=torch.tensor(teacher_ouputs["hidden_states"][30::3,-1], dtype=torch.float32)

        return teacher_ouputs
