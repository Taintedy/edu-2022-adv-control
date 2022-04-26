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
    def __init__(self, action_predefined, objects=[], pars=[]):
        super().__init__(objects, pars)
        self.full_trajectory = []
        self.action_predefined = action_predefined

    def animate(self, k):
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            state_full = self.state_full
            action = self.action
            stage_obj = self.stage_obj
            accum_obj = self.accum_obj

        else:
            self.simulator.sim_step()

            t, state, observation, state_full = self.simulator.get_sim_step_data()

            action = self.ctrl_selector(
                t,
                observation,
                self.action_predefined,
                self.action_manual,
                self.ctrl_nominal,
                self.ctrl_benchmarking,
                self.ctrl_mode,
            )

            self.sys.receive_action(action)
            self.ctrl_benchmarking.receive_sys_state(self.sys._state)
            self.ctrl_benchmarking.upd_accum_obj(observation, action)

            stage_obj = self.ctrl_benchmarking.stage_obj(observation, action)
            accum_obj = self.ctrl_benchmarking.accum_obj_val

        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        alpha_deg = alpha / np.pi * 180

        self.full_trajectory.append([xCoord, yCoord, alpha_deg])

        if self.is_print_sim_step:
            self.logger.print_sim_step(
                t, xCoord, yCoord, alpha, stage_obj, accum_obj, action
            )

        if self.is_log_data:
            self.logger.log_data_row(
                self.datafile_curr,
                t,
                xCoord,
                yCoord,
                alpha,
                stage_obj,
                accum_obj,
                action,
            )

        # xy plane
        text_time = "t = {time:2.3f}".format(time=t)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot

        self.robot_marker.rotate(1e-3)  # Rotate the robot on the plot
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(
            5, 5, marker=self.robot_marker.marker, s=400, c="b"
        )

        self.robot_marker.rotate(alpha_deg)  # Rotate the robot on the plot
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(
            xCoord, yCoord, marker=self.robot_marker.marker, s=400, c="b"
        )

        # # Solution
        upd_line(self.line_norm, t, la.norm([xCoord, yCoord]))
        upd_line(self.line_alpha, t, alpha)

        # Cost
        upd_line(self.line_stage_obj, t, stage_obj)
        upd_line(self.line_accum_obj, t, accum_obj)
        text_accum_obj = r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}".format(
            accum_obj=accum_obj
        )
        upd_text(self.text_accum_obj_handle, text_accum_obj)

        # Control
        for (line, action_single) in zip(self.lines_ctrl, action):
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
            reset_line(self.lines_ctrl[0])
            reset_line(self.lines_ctrl[1])

            upd_line(self.line_traj, np.nan, np.nan)


def generate_data_for_task(cost_calculator_instance):

    globals().update(vars((update_globals_for_test())))

    action_sqn_multiplier = 2
    sin_action_sqn = action_sqn_multiplier * np.sin(
        np.linspace(1, 5, cost_calculator_instance.Nactor * 2)
    )
    lin_action_sqn = action_sqn_multiplier * np.linspace(
        1, 5, cost_calculator_instance.Nactor * 2
    )
    const_action_sqn = action_sqn_multiplier * np.ones(
        cost_calculator_instance.Nactor * 2
    )
    sin_action_sqn[1::2] = sin_action_sqn[1::2] * action_sqn_multiplier ** 3
    lin_action_sqn[1::2] = lin_action_sqn[1::2] * action_sqn_multiplier ** 2
    const_action_sqn[1::2] = (
        sin_action_sqn[1::2] * action_sqn_multiplier ** 2
    )
    first_observation = [2, 2, 0.8]
    action_list = [sin_action_sqn, lin_action_sqn, const_action_sqn]
    observation_list = []
    for action_sqn in action_list:
        cost_calculator_instance._actor_cost(action_sqn, first_observation)
        observation_list.append(cost_calculator_instance.observation_sqn)
    return observation_list


def test_first_task_procedure(
    cost_calculator_instance, ref_observation_sequences, tol=1e-2
):
    first_observation = [2, 2, 0.8]
    action_sqn_multiplier = 2

    ref_robot_marker = visuals.RobotMarker(angle=first_observation[2])
    test_robot_marker = visuals.RobotMarker(angle=first_observation[2])

    sin_action_sqn = action_sqn_multiplier * np.sin(
        np.linspace(1, 5, cost_calculator_instance.Nactor * 2)
    )
    lin_action_sqn = action_sqn_multiplier * np.linspace(
        1, 5, cost_calculator_instance.Nactor * 2
    )
    const_action_sqn = action_sqn_multiplier * np.ones(
        cost_calculator_instance.Nactor * 2
    )

    sin_action_sqn[1::2] = sin_action_sqn[1::2] * action_sqn_multiplier ** 3
    lin_action_sqn[1::2] = lin_action_sqn[1::2] * action_sqn_multiplier ** 2
    const_action_sqn[1::2] = sin_action_sqn[1::2] * action_sqn_multiplier ** 2
    action_list = [sin_action_sqn, lin_action_sqn, const_action_sqn]

    metrics = {}
    metrics_names = [
        "$l_1$-difference metric",
        "$l_2$-difference metric",
        "$l_\infty$-difference metric",
    ]
    action_type = [
        "$\quad$ Sinusoidal $\quad$",
        "$\quad$ Linear $\quad$",
        "$\quad$ Constant $\quad$",
    ]

    plt.subplots(1, len(ref_observation_sequences), figsize=(10, 3))
    for i in range(1, len(ref_observation_sequences) + 1):
        cost_calculator_instance._actor_cost(action_list[i - 1], first_observation)
        test_observation_sqn = cost_calculator_instance.observation_sqn[1:, :]
        ref_observation_sqn = ref_observation_sequences[i - 1][1:, :]
        cur_metrics = [
            np.linalg.norm(test_observation_sqn - ref_observation_sqn, 1),
            np.linalg.norm(test_observation_sqn - ref_observation_sqn),
            np.linalg.norm(test_observation_sqn - ref_observation_sqn, np.inf),
        ]

        metrics[action_type[i - 1]] = cur_metrics
        plt.subplot(1, len(ref_observation_sequences), i)
        for j in range(test_observation_sqn.shape[0]):
            test_robot_marker.rotate(test_observation_sqn[j, 2] / np.pi * 180)
            ref_robot_marker.rotate(ref_observation_sqn[j, 2] / np.pi * 180)
            if j == 0:
                plt.scatter(
                    test_observation_sqn[j, 0],
                    test_observation_sqn[j, 1],
                    marker=test_robot_marker.marker,
                    s=80.1,
                    c="b",
                    label="Your predictions",
                )
                plt.scatter(
                    ref_observation_sqn[j, 0],
                    ref_observation_sqn[j, 1],
                    marker=ref_robot_marker.marker,
                    s=80.1,
                    alpha=0.5,
                    c="r",
                    label="Reference predictions",
                )
            else:
                plt.scatter(
                    test_observation_sqn[j, 0],
                    test_observation_sqn[j, 1],
                    marker=test_robot_marker.marker,
                    s=80.1,
                    c="b",
                )
                plt.scatter(
                    ref_observation_sqn[j, 0],
                    ref_observation_sqn[j, 1],
                    marker=ref_robot_marker.marker,
                    s=80.1,
                    alpha=0.5,
                    c="r",
                )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=1,
            fancybox=True,
            shadow=True,
        )
        plt.axis("equal")
    final_report = pd.DataFrame(metrics, index=metrics_names)
    display(final_report)
    to_grade = np.max(final_report.values)
    if to_grade <= tol:
        print(
            f"max difference between reference and predicted observations equals {to_grade} and small enough. Success!"
        )
        return True
    else:
        print(
            f"max difference between reference and predicted observations equals {to_grade} and too big."
        )
        return False


def integral_grading(trajectory):
    tail_length = len(trajectory) // 5
    tail = np.array(trajectory)[-tail_length:, :2]
    norms = la.norm(tail, axis=1)
    integral_metric = norms.sum() * 0.1
    plt.figure(figsize=(4, 4))
    plt.plot(norms, label="norm of radius-vector")
    plt.xlabel("time")
    plt.grid()
    plt.legend()
    print(abs(integral_metric - 1.318) / 1.318 )
    return abs(integral_metric - 1.318) / 1.318


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
                    default=0.01,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=10.0,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                    default=['5', '5', '-3*pi/4', '0', '0'],
                    help='Initial state (as sequence of numbers); ' + 
                    'dimension is environment-specific!')
parser.add_argument('--is_log_data', type=str2bool,
                    default=False,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=str2bool,
                    default=True,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=str2bool,
                    default=True,
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
        F_sign = np.sign(-np.cos(alpha)/np.sqrt(x**2 + y**2)-10*np.cos(alpha)*(x**2 + y**2))
        F_abs = abs(-np.cos(alpha)/np.sqrt(x**2 + y**2)-10*np.cos(alpha)*(x**2 + y**2))

        M_sign = np.sign(3* alpha *np.sin(3*t))
        M_abs = abs(3* alpha *np.sin(3*t))

        F = F_sign * min(F_abs, 300)
        M = M_sign * min(M_abs, 100)
        return [F, M]



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

    args.my_simulator = simulator.Simulator(sys_type = "diff_eqn",
                                   closed_loop_rhs = args.my_sys.closed_loop_rhs,
                                   sys_out = args.my_sys.out,
                                   state_init = args.state_init,
                                   disturb_init = [],
                                   action_init = args.action_init,
                                   t0 = args.t0,
                                   t1 = args.t1,
                                   dt = args.dt,
                                   max_step = args.dt/2,
                                   first_step = 1e-6,
                                   atol = args.atol,
                                   rtol = args.rtol,
                                   is_disturb = args.is_disturb,
                                   is_dyn_ctrl = args.is_dyn_ctrl)
    args.datafiles = [None] * args.Nruns
    args.my_logger = loggers.Logger3WRobot()

    return args