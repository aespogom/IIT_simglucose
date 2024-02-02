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
    def __init__(self):
        super().__init__()
        # Oracle simulator
        self.model = self.simulator
        self.loss = nn.MSELoss()
        # specify start_time as the beginning of today
        now = datetime.now()
        self.start_time = datetime.combine(now.date(), datetime.min.time())

    def simulator(self, index, pat_name):
        # Create a simulation environment
        patient = T1DPatient.withName(name=pat_name)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        # custom scenario is a list of tuples (time, meal_size)
        scen = [(7, 45), (12, 70), (16, 15), (18, 80), (23, 10)]
        scenario = CustomScenario(start_time=self.start_time, scenario=scen)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()

        controller.reset()
        obs, reward, done, info = env.reset()
        while env.time < scenario.start_time + timedelta(days=1):
            action = controller.policy(obs, reward, done, **info)
            obs, reward, done, info = env.step(action)
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
            input_ids: patient names
            labels: real labels
            interchanged_variables: alignment, 
            variable_names: mapping,
            interchanged_activations: values to interchange
        """

        teacher_ouputs = {}
        teacher_ouputs["hidden_states"]=[]
        # we perform the interchange intervention
        for i, pat_name in enumerate(input_ids):
            x = self.simulator(i, pat_name)
            # we need to interchange!
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    x = interchanged_activations
            
            teacher_ouputs["hidden_states"].append(x)
        
        tensor_preds = torch.zeros((1,3))
        ## TODO COMPLETE WITH 3 TIMESERIES PREDICTION
        teacher_ouputs["outputs"]=tensor_preds

        return teacher_ouputs
