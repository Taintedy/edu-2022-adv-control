import numpy as np

from rcognita import simulator
from rcognita import systems
from rcognita import controllers
from rcognita import loggers
from rcognita import visuals
from rcognita.utilities import on_key_press
from utilities import dss_sim
from utilities import rep_mat
from utilities import uptria2vec
from utilities import push_vec
from argparser_3wrobot_NI import parser
from asgn_2_stub import CtrlOptPredBase
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


description = "Agent-environment preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=['manual',
                             'nominal',
                             'MPC',
                             'RQL',
                             'SQL',
                             'JACS'],
                    default='nominal',
                    help='Control mode. Currently available: ' +
                    '----manual: manual constant control specified by action_manual; ' +
                    '----nominal: nominal controller, usually used to benchmark optimal controllers;' +                     
                    '----MPC:model-predictive control; ' +
                    '----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; ' +
                    '----SQL: stacked Q-learning; ' + 
                    '----JACS: joint actor-critic (stabilizing), system-specific, needs proper setup.')
parser.add_argument('--dt', type=float, metavar='dt',
                    default=0.01,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=10.0,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                    default=['5', '5', '-3*pi/4'],
                    help='Initial state (as sequence of numbers); ' + 
                    'dimension is environment-specific!')
parser.add_argument('--is_log_data', type=str2bool,
                    default=False,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=str2bool, nargs='?',
                        const=True, default=True,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=str2bool,
                    default=True,
                    help='Flag to print simulation data into terminal.')
parser.add_argument('--prob_noise_pow', type=float,
                    default=False,
                    help='Power of probing (exploration) noise.')
parser.add_argument('--action_manual', type=float,
                    default=[-5, -3], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int,
                    default=3,
                    help='Horizon length (in steps) for predictive controllers.')
parser.add_argument('--pred_step_size_multiplier', type=float,
                    default=1.0,
                    help='Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.')
parser.add_argument('--stage_obj_struct', type=str,
                    default='quadratic',
                    choices=['quadratic',
                             'biquadratic'],
                    help='Structure of stage objective function.')
parser.add_argument('--R1_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0],
                    help='Parameter of stage objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--R2_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0],
                    help='Parameter of stage objective function . Must have proper dimension. ' + 
                    'Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, ' +
                    'where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--gamma', type=float,
                    default=1.0,
                    help='Discount factor.')
parser.add_argument('--actor_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix'],
                    help='Feature structure (actor). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms.')



class CtrlOptPredBase:         
    def __init__(self,
                 dim_input,
                 dim_output,
                 ctrl_bnds=[],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 prob_noise_pow = 1,
                 gamma=1,
                 stage_obj_struct='quadratic',
                 stage_obj_pars=[],
                 observation_target=[]):
        
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size

        self.stage_obj_struct = stage_obj_struct
        self.stage_obj_pars = stage_obj_pars
        self.observation_target = observation_target
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        
        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)       
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys        
        
        self.accum_obj_val = 0


    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.
        
        """
        self.ctrl_clock = t0
        self.action_curr = self.action_min/10
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def stage_obj(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        stage_obj = 0

        if self.stage_obj_struct == 'quadratic':
            R1 = self.stage_obj_pars[0]
            stage_obj = chi @ R1 @ chi
        elif self.stage_obj_struct == 'biquadratic':
            R1 = self.stage_obj_pars[0]
            R2 = self.stage_obj_pars[1]
            stage_obj = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
        
        return stage_obj
        
    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """
        self.accum_obj_val += self.stage_obj(observation, action)*self.sampling_time
        
                    
    def compute_action(self, t, observation): 
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            action = self._actor_optimizer(observation)
            self.action_curr = action
            
            return action    
    
        else:
            return self.action_curr