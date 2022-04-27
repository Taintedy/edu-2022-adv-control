from math import ceil
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
from rcognita.rcognita import visuals, controllers, systems
from IPython.display import display, Markdown
from rcognita.rcognita.utilities import upd_line
from rcognita.rcognita.utilities import reset_line
from rcognita.rcognita.utilities import upd_text
from rcognita.rcognita import simulator
from rcognita.rcognita import loggers
import numpy.linalg as la

class Animator3WRobot_traj(visuals.Animator3WRobot):
    def __init__(self, objects=[], pars=[]):
        super().__init__(objects, pars)
        self.full_trajectory = []
        self.distance = 0
        self.last_time = 0
        self.est_full_trajectory = []
        self.noisy_observ_buffer = []
        self.key_points = np.arange(1,10)
        self.last_time_snapshot = 0
        self.t_array = []

    def animate(self, k):
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            state_full = self.state_full
            action = self.action
            #stage_obj = self.stage_obj
            accum_obj = self.accum_obj

        else:
            self.simulator.sim_step()

            t, state, observation, state_full = self.simulator.get_sim_step_data()
          
            action = self.ctrl_selector(
                t,
                observation,
                self.action_manual,
                self.ctrl_nominal,
                self.ctrl_benchmarking,
                self.ctrl_mode,
            )
            

            self.sys.receive_action(action)
            self.ctrl_benchmarking.receive_sys_state(self.sys._state)
            self.ctrl_benchmarking.upd_accum_obj(observation, action)

            #stage_obj = self.ctrl_benchmarking.stage_obj(observation, action)
            accum_obj = self.ctrl_benchmarking.accum_obj_val
        self.t_array.append(t)
        delta_time = t - self.last_time
        self.last_time = t
        if len(self.full_trajectory) > 0:
            self.distance += delta_time * self.full_trajectory[-1][3]

        noisy_observation = observation +  np.random.randn(len(observation)) 
        self.noisy_observ_buffer.append(noisy_observation)
        est_state_full = self.state_estimator.compute_estimate(t, noisy_observation, action)
        self.full_trajectory.append(state_full)
        self.est_full_trajectory.append(est_state_full)

        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        alpha_deg = alpha / np.pi * 180
        v = state_full[3]
        omega = state_full[4]
        
        est_xCoord = est_state_full[0]
        est_yCoord = est_state_full[1]
        est_alpha = est_state_full[2]

        est_alpha_deg = est_alpha / np.pi * 180
        
        stage_obj = la.norm((state_full-est_state_full)[:3])

        if self.is_print_sim_step:
            self.logger.print_sim_step(
                t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action
            )

        if self.is_log_data:
            self.logger.log_data_row(
                self.datafile_curr,
                t,
                xCoord,
                yCoord,
                alpha,
                v,
                omega,
                stage_obj,
                accum_obj,
                action,
            )

        # xy plane
        text_time = "t = {time:2.3f}".format(time=t)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot
        upd_line(self.line_traj_est, est_xCoord, est_yCoord)

        self.robot_marker.rotate(alpha_deg)  # Rotate the robot on the plot
        self.robot_marker_est.rotate(est_alpha_deg) 
        if self.distance > 2.5:
            self.distance = 0
            self.scatter_sol_snapshot = self.axs_xy_plane.scatter(
            xCoord, yCoord, marker=self.robot_marker.marker, s=400, c="b", 
            )
            self.scatter_sol_snapshot_est = self.axs_xy_plane.scatter(
            est_xCoord, est_yCoord, marker=self.robot_marker_est.marker, s=400, c="r"
            )
            self.last_time_snapshot = t


        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(
            xCoord, yCoord, marker=self.robot_marker.marker, s=400, c="b", label="ground truth"
        )
        self.scatter_sol_est.remove()
        self.scatter_sol_est = self.axs_xy_plane.scatter(
            est_xCoord, est_yCoord, marker=self.robot_marker_est.marker, s=400, c="r", label="estimation"
        )
        self.axs_xy_plane.legend(borderpad=0.9, labelspacing=1.8, bbox_to_anchor=(1., 1.), loc='upper right')
        # # Solution
        #upd_line(self.line_norm, t, la.norm([xCoord, yCoord]))
        
        #upd_line(self.line_alpha, t, alpha)
        
        upd_line(self.my_line, t, v)
        #upd_line(self.my_line_est, t, est_v)
        
        upd_line(self.my_line2, t, omega)
        #upd_line(self.my_line2_est, t, est_omega)
        
        
        

        # Cost
        upd_line(self.line_stage_obj, t, stage_obj)
        #upd_line(self.line_accum_obj, t, accum_obj)
        #text_accum_obj = r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}".format(
        #    accum_obj=accum_obj
        #)
        #upd_text(self.text_accum_obj_handle, text_accum_obj)
        
        # Control
        for (line, action_single) in zip(self.lines_coord, [xCoord, yCoord, alpha]):
            upd_line(line, t, action_single)

        for (line, action_single) in zip(self.est_lines_coord, [est_xCoord, est_yCoord, est_alpha]):
            upd_line(line, t, action_single)

        # Run done
        if t >= self.t1:
            if self.is_print_sim_step:
                print(
                    ".....................................Run {run:2d} done.....................................".format(
                        run=self.run_curr
                    )
                )

            self.run_curr += 1

            if self.run_curr > self.Nruns:
                print("Animation done...")
                self.stop_anm()
                return

            if self.is_log_data:
                self.datafile_curr = self.datafiles[self.run_curr - 1]

            # Reset simulator
            self.simulator.reset()

            # Reset controller
            if self.ctrl_mode > 0:
                self.ctrl_benchmarking.reset(self.t0)
            else:
                self.ctrl_nominal.reset(self.t0)

            accum_obj = 0

            reset_line(self.line_norm)
            reset_line(self.line_alpha)
            reset_line(self.line_stage_obj)
            reset_line(self.line_accum_obj)
            reset_line(self.lines_coord[0])
            reset_line(self.lines_coord[1])

            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)

            upd_line(self.line_traj, np.nan, np.nan)





def test_first_task_procedure(
    sys_jacobi_func, 
    sys_observ_func
):
    TEST_VECTORS_ARRAY = np.array([
    [0.61802688, 0.98055101, 0.39646692, 0.83884212, 0.05085759],
    [0.95042846, 0.98278097, 0.20653316, 0.5615744 , 0.90778497],
    [0.69384927, 0.60517308, 0.56655476, 0.21288348, 0.85700927],
    [0.93903704, 0.75367938, 0.65446551, 0.60976157, 0.44653094],
    [0.80372913, 0.26904169, 0.35050789, 0.72469409, 0.10591367],
    ])

    TEST_JACOBI_MATRIX = np.array([
    [2.23606798, 0.        , 0.05752796, 0.20080548, 0.        ],
    [0.        , 2.23606798, 0.12768654, 0.09837256, 0.        ],
    [0.        , 0.        , 2.23606798, 0.        , 0.2236068 ],
    [0.        , 0.        , 0.        , 2.23606798, 0.        ],
    [0.        , 0.        , 0.        , 0.        , 2.23606798],
    ])

    result_matrix = np.apply_along_axis(
        sys_jacobi_func, 
        axis=1, 
        arr=TEST_VECTORS_ARRAY
    )

    result_matrix = np.apply_along_axis(np.linalg.norm, axis=0, arr=result_matrix)

    error_sys_jacobi = la.norm(TEST_JACOBI_MATRIX - result_matrix)

    error_det = la.det(sys_observ_func() @ sys_observ_func().T) - 1

    error_shape = sys_observ_func().shape == (2,5)

    condition_1 = (error_sys_jacobi < 2e-9)
    condition_2 = (error_det == 0.)
    condition_3 = (error_shape)
    if (error_sys_jacobi < 1e-8) & (error_det == 0.) & (error_shape):
        display(Markdown(f"<text style=color:blue> Great job! Grade is 25 out of 25. </text>"))
        grade = 25
    else:
        cond_array = np.array([condition_1, condition_2, condition_3])
        indicies_of_errors = np.array(np.where(cond_array==False))[0] + 1
        indicies_of_errors = [str(x) for x in indicies_of_errors]
        indicies_of_errors = ', '.join(indicies_of_errors)
        display(Markdown(f"<text style=color:red> Something went wrong.\
         Check out criteria {indicies_of_errors}. Grade is 0 out of 25.</text>"))
        grade = 0
    
    return grade

    


def integral_grading(trajectory):
    tail_length = len(trajectory) - 2
    tail = np.array(trajectory)[-tail_length:, :2]
    norms = la.norm(tail, axis=1)
    integral_metric = norms.sum() * 0.01
    result = (integral_metric - 33) / 33
    if result < 0.05:
        display(Markdown(f"<text style=color:blue> Perfect! Grade is 75 out of 75. </text>"))
        grade = 75
    elif (result > 0.05) & (result < 0.2):
        display(Markdown(f"<text style=color:green> Not bad :) Grade is {int((1 - result)*75)} out of 75. </text>"))
        grade = 1 - result
    else:
        display(Markdown(f"<text style=color:red> Try more! Grade is 0 out of 75. </text>"))
        grade = 0
    return grade


def final_grade(g1, g2):
    final_grade = (g1 == 1) * 60 + ceil(max((1 - g2),0) * 40) 
    display(Markdown(f"<strong> Your grade is: {final_grade}</strong>"))
    if final_grade > 95:
        display(Markdown(f"<text style=color:blue> Perfect! </text>"))
    elif final_grade > 90:
        display(Markdown(f"<text style=color:green> Good. </text>"))
    elif final_grade > 85:
        display(Markdown(f"<text style=color:brown> Acceptable :) </text>"))
    if final_grade < 85:
        display(Markdown(f"<text style=color:red>Looks like one could do better!</text>"))
        #print("\x1b[31m Minimal value for passing is 85\x1b[0m")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


description = "Agent-environment preset: 3-wheel robot with dynamical actuators."

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=[
                             'manual',
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
                    default=0.005,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=10.0,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                    default=['-5', '0', '-3*pi/4', '0', '0.7'],
                    help='Initial state (as sequence of numbers); ' + 
                    'dimension is environment-specific!')
parser.add_argument('--is_log_data', type=str2bool,
                    default=False,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=str2bool,
                    default=True,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=str2bool,
                    default=False,
                    help='Flag to print simulation data into terminal.')
parser.add_argument('--is_est_model', type=str2bool,
                    default=False,
                    help='Flag to estimate environment model.')
parser.add_argument('--model_est_stage', type=float,
                    default=1.0,
                    help='Seconds to learn model until benchmarking controller kicks in.')
parser.add_argument('--model_est_period_multiplier', type=float,
                    default=1,
                    help='Model is updated every model_est_period_multiplier times dt seconds.')
parser.add_argument('--model_order', type=int,
                    default=5,
                    help='Order of state-space estimation model.')
parser.add_argument('--prob_noise_pow', type=float,
                    default=False,
                    help='Power of probing (exploration) noise.')
parser.add_argument('--action_manual', type=float,
                    default=[-5, -3], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int,
                    default=5,
                    help='Horizon length (in steps) for predictive controllers.')
parser.add_argument('--pred_step_size_multiplier', type=float,
                    default=2.0,
                    help='Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.')
parser.add_argument('--buffer_size', type=int,
                    default=10,
                    help='Size of the buffer (experience replay) for model estimation, agent learning etc.')
parser.add_argument('--stage_obj_struct', type=str,
                    default='quadratic',
                    choices=['quadratic',
                             'biquadratic'],
                    help='Structure of stage objective function.')
parser.add_argument('--R1_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0, 0, 0],
                    help='Parameter of stage objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--R2_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0, 0, 0],
                    help='Parameter of stage objective function . Must have proper dimension. ' + 
                    'Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, ' +
                    'where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--Ncritic', type=int,
                    default=4,
                    help='Critic stack size (number of temporal difference terms in critic cost).')
parser.add_argument('--gamma', type=float,
                    default=1.0,
                    help='Discount factor.')
parser.add_argument('--critic_period_multiplier', type=float,
                    default=1.0,
                    help='Critic is updated every critic_period_multiplier times dt seconds.')
parser.add_argument('--critic_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix',
                                                   'quad-mix'],
                    help='Feature structure (critic). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms; ' +
                    '----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations).')
parser.add_argument('--actor_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix'],
                    help='Feature structure (actor). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms.')

args = parser.parse_args([])

class CtrlPredefined3WRobot(controllers.CtrlNominal3WRobot):
    def compute_action(self, t, observation):
        x = observation[0]
        y = observation[1]
        alpha = observation[2]
        v = observation[3]
        omega = observation[4]
        # F_sign = np.sign(-np.cos(alpha)/np.sqrt(x**2 + y**2)-10*np.cos(alpha)*(x**2 + y**2))
        # F_abs = abs(-np.cos(alpha)/np.sqrt(x**2 + y**2)-10*np.cos(alpha)*(x**2 + y**2))

        # M_sign = np.sign(3* alpha *np.sin(3*t))
        # M_abs = abs(3* alpha *np.sin(3*t))

        F = 300 if v < 6 else 0
        M = 56*np.cos(t*9)#2  if int(t * 1.5) % 2 ==0 else -2#if int((t+1)) % 3 == 0 else 3*np.sin(t) #if omega > -5 else 5#*np.sin(t*5) * np.sign(np.cos(alpha))
        return [F, M]
    # def compute_action(self, t, observation):

    #     F = 5
    #     M = 0
    #     return [F, M]



def update_globals_for_test(args=args):

    if isinstance(args.state_init[0], str):
        for k in range(len(args.state_init)):
            args.state_init[k] = eval(args.state_init[k].replace("pi", str(np.pi)))

    args.state_init = np.array(args.state_init)
    args.action_manual = np.array(args.action_manual)
    dim_state = 5
    args.dim_state = dim_state
    args.state_init = np.array(args.state_init)
    args.action_manual = np.array(args.action_manual)
    args.dim_state = dim_state
    args.dim_input = 2
    args.dim_output = args.dim_state
    args.dim_disturb = 0

    args.dim_R1 = args.dim_output + args.dim_input
    args.dim_R2 = args.dim_R1
    args.pred_step_size = args.dt * args.pred_step_size_multiplier
    args.model_est_period = args.dt * args.model_est_period_multiplier
    args.critic_period = args.dt * args.critic_period_multiplier

    args.R1 = np.diag(np.array(args.R1_diag))
    args.R2 = np.diag(np.array(args.R2_diag))
    assert args.t1 > args.dt > 0.0
    assert args.state_init.size == args.dim_state

    args.is_disturb = 0
    args.is_dyn_ctrl = 0

    args.t0 = 0

    args.action_init = 0 * np.ones(args.dim_input)

    # Solver
    args.atol = 1e-5
    args.rtol = 1e-3

    # xy-plane
    args.xMin = -10
    args.xMax = 10
    args.yMin = -10
    args.yMax = 10

    # Model estimator stores models in a stack and recall the best of model_est_checks
    args.model_est_checks = 0

    # Control constraints
    args.Fmin = -300
    args.Fmax = 300
    args.Mmin = -100
    args.Mmax = 100
    args.ctrl_bnds = np.array([[args.Fmin, args.Fmax], [args.Mmin, args.Mmax]])

    # System parameters
    args.m = 10  # [kg]
    args.I = 1  # [kg m^2]

#     args.my_ctrl_nominal = controllers.CtrlNominal3WRobot(
#     args.m, args.I, ctrl_gain=5, ctrl_bnds=args.ctrl_bnds, t0=args.t0, sampling_time=args.dt
# )
    args.my_ctrl_nominal = CtrlPredefined3WRobot(
    args.m, args.I, ctrl_gain=5, ctrl_bnds=args.ctrl_bnds, t0=args.t0, sampling_time=args.dt
)
    args.my_sys = systems.Sys3WRobot(
        sys_type="diff_eqn",
        dim_state=args.dim_state,
        dim_input=args.dim_input,
        dim_output=args.dim_output,
        dim_disturb=args.dim_disturb,
        pars=[args.m, args.I],
        ctrl_bnds=args.ctrl_bnds,
        is_dyn_ctrl=args.is_dyn_ctrl,
        is_disturb=args.is_disturb,
        pars_disturb=[],
    )

    args.observation_init = args.my_sys.out(args.state_init)
    args.my_ctrl_opt_pred = controllers.CtrlOptPred(
        args.dim_input,
        args.dim_output,
        args.ctrl_mode,
        ctrl_bnds=args.ctrl_bnds,
        action_init=[],
        t0=args.t0,
        sampling_time=args.dt,
        Nactor=args.Nactor,
        pred_step_size=args.pred_step_size,
        sys_rhs=args.my_sys._state_dyn,
        sys_out=args.my_sys.out,
        state_sys=args.state_init,
        prob_noise_pow=args.prob_noise_pow,
        is_est_model=args.is_est_model,
        model_est_stage=args.model_est_stage,
        model_est_period=args.model_est_period,
        buffer_size=args.buffer_size,
        model_order=args.model_order,
        model_est_checks=args.model_est_checks,
        gamma=args.gamma,
        Ncritic=args.Ncritic,
        critic_period=args.critic_period,
        critic_struct=args.critic_struct,
        stage_obj_struct=args.stage_obj_struct,
        stage_obj_pars=[args.R1],
        observation_target=[],
    )
    args.my_ctrl_benchm = args.my_ctrl_opt_pred
    args.xCoord0 = args.state_init[0]
    args.yCoord0 = args.state_init[1]
    args.alpha0 = args.state_init[2]
    args.alpha_deg_0 = args.alpha0/2/np.pi
    args.max_step = args.dt/2

    args.my_simulator = simulator.Simulator(sys_type = "diff_eqn",
                                   closed_loop_rhs = args.my_sys.closed_loop_rhs,
                                   sys_out = args.my_sys.out,
                                   state_init = args.state_init,
                                   disturb_init = [],
                                   action_init = args.action_init,
                                   t0 = args.t0,
                                   t1 = args.t1,
                                   dt = args.dt,
                                   max_step = args.max_step,
                                   first_step = 1e-6,
                                   atol = args.atol,
                                   rtol = args.rtol,
                                   is_disturb = args.is_disturb,
                                   is_dyn_ctrl = args.is_dyn_ctrl)
    args.datafiles = [None] * args.Nruns
    args.my_logger = loggers.Logger3WRobot()

    return args