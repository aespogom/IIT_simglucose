import os
from simglucose.patient.base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ['CHO', 'insulin'])
Observation = namedtuple("observation", ['Gsub'])

PATIENT_PARA_FILE = os.path.join('data', 'sampled_insilico_vparams.csv')
PATIENT_PARA_FILE_TEST = os.path.join('data', 'insilico_vparams.csv')


class T1DPatient(Patient):
    SAMPLE_TIME = 3  # min
    EAT_RATE = 5  # g/min CHO

    def __init__(self,
                 params,
                 init_state=None,
                 random_init_bg=False,
                 seed=None,
                 t0=0):
        '''
        T1DPatient constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
        '''
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0
        self.state_hist = []
        self.reset()

    @classmethod
    def withID(cls, patient_id, **kwargs):
        '''
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        '''
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        '''
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        '''
        if "test" in name:
            patient_params = pd.read_csv(PATIENT_PARA_FILE_TEST)
            params = patient_params.loc[patient_params.Name == name].squeeze()
        else:
            patient_params = pd.read_csv(PATIENT_PARA_FILE)
            params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        current_state = self._odesolver.y
        # DEFINITION CURRENT STATE --> LOCALITY OF INTERCHANGE
        ## dxdt1 --> Stomach solid
        ## dxdt2 --> Stomach liquid
        ## dxdt3 --> Intestine
        ## dxdt4 --> Glucose kinetics
        ## dxdt5 --> Glucose kinetics
        ## dxdt6 --> Insulin kinetics
        ## dxdt7 --> Insulin action on glucose utilization
        ## dxdt8 --> Insulin action on production
        ## dxdt9 --> Insulin action on production
        ## dxdt10 --> Insulin in the liver (pmol/kg)
        ## dxdt11 --> Subcutaneous insulin kinetics
        ## dxdt12 --> Subcutaneous insulin kinetics
        ## dxdt13 --> Subcutaneous glucose --> "ground truth"
        return current_state

    @property
    def t(self):
        return int(self._odesolver.t)

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action,
            pred_horizon,
            interchanged_variables=None,
            variable_names=None,
            interchanged_activations=None,
            timeseries_iit=False):
        self._odesolver.t = int(self._odesolver.t)
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            # logger.info('t = {}, patient starts eating ...'.format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True

        if to_eat > 0:
            logger.debug('t = {}, patient eats {} g'.format(
                self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            # logger.info('t = {}, Patient finishes eating!'.format(self.t))
            self.is_eating = False

        # Update last input
        self._last_action = action

        save_state = None
        # IIT for MLP scaled
        if variable_names != None and timeseries_iit:
            dict_mapping = {}
            start = 0
            stop = 13
            for last_minute in range(30, 30+pred_horizon+1, self.sample_time):
                dict_mapping[str(last_minute)] = range(start, stop)
                start = stop
                stop = stop+13
            assert interchanged_variables != None
            for param in variable_names:
                if str(int(self._odesolver.t)) in dict_mapping and param in dict_mapping[str(int(self._odesolver.t))]:
                    interchanged_activations = interchanged_variables[0]
                    interchanged_state = self.state
                    save_state = self.state.copy()
                    index_param = dict_mapping[str(int(self._odesolver.t))].index(param)
                    interchanged_state[index_param] = interchanged_activations
                    self._odesolver.set_initial_value(interchanged_state, self._odesolver.t)
        elif variable_names != None and int(self._odesolver.t)==30+pred_horizon-1 and not timeseries_iit:
            # IIT for MLP tree or MLP parallel
            assert interchanged_variables != None
            for param in variable_names:
                interchanged_activations = interchanged_variables[0]
                interchanged_state = self.state
                save_state = self.state.copy()
                interchanged_state[param] = interchanged_activations
                self._odesolver.set_initial_value(interchanged_state, self._odesolver.t)

        self._odesolver.set_f_params(action, self._params, self._last_Qsto,
                                     self._last_foodtaken)
        if self._odesolver.successful():
            try:
                t_next = self._odesolver.t + self.sample_time
                self._odesolver.integrate(t_next)
                self.state_hist.append(self.state)
            except Exception as e:
                logger.info(f'ODE solver failed!!: {e}')
                logger.info(self._odesolver.y)
                logger.info(self._odesolver.get_return_code())
                if interchanged_variables is not None and save_state is not None:
                    logger.info('Revert interchange intervention')
                    logger.info(save_state)
                    self._odesolver.set_initial_value(save_state, int(self._odesolver.t))
                    self._odesolver.integrate(t_next)
                    pass
                else:
                    logger.error('No solution :(')


    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        dxdt = np.zeros(13)
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # Glucose in the stomach
        qsto = x[0] + x[1]
        # NOTE: Dbar is in unit mg, hence last_foodtaken needs to be converted
        # from mg to g. See https://github.com/jxx123/simglucose/issues/41 for
        # details.
        Dbar = last_Qsto + last_foodtaken * 1000  # unit: mg

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar)) -
                np.tanh(cc * (qsto - params.d * Dbar)) + 2)
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - \
            params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = -(params.m2 + params.m4) * x[5] + params.m1 * x[9] + params.ka1 * \
            x[10] + params.ka2 * x[11]  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        dxdt[12] = (-params.ksc * x[12] + params.ksc * x[3])
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        if action.insulin > basal:
            logger.debug('t = {}, injecting insulin: {}'.format(
                t, action.insulin))

        return dxdt

    @property
    def observation(self):
        '''
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        '''
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation

    def _announce_meal(self, meal):
        '''
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        '''
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    def reset(self):
        '''
        Reset the patient state to default intial state
        '''
        if self._init_state is None:
            self.init_state = self._params.iloc[2:15]
        else:
            self.init_state = self._init_state

        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            # Only randomize glucose related states, x4, x5, and x13
            mean = [
                1.0 * self.init_state[3], 1.0 * self.init_state[4],
                1.0 * self.init_state[12]
            ]
            cov = np.diag([
                0.1 * self.init_state[3], 0.1 * self.init_state[4],
                0.1 * self.init_state[12]
            ])
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3] = 1.0 * bg_init[0]
            self.init_state[4] = 1.0 * bg_init[1]
            self.init_state[12] = 1.0 * bg_init[2]

        self._last_Qsto = self.init_state.iloc[0] + self.init_state.iloc[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._odesolver = ode(self.model).set_integrator('dopri5')
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0

