from math import ceil
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
from rcognita.rcognita import visuals
from IPython.display import display, Markdown
import html
from rcognita.rcognita.utilities import upd_line
from rcognita.rcognita.utilities import reset_line
from rcognita.rcognita.utilities import upd_text
import numpy.linalg as la


class Animator3WRobotNI_traj(visuals.Animator3WRobotNI):
    def __init__(self, objects=[], pars=[]):
        super().__init__(objects, pars)
        self.full_trajectory = []
        self.observ_state_diff = []

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

            self.ctrl_benchmarking.receive_sys_state(self.sys._state)

            action = self.ctrl_selector(
                t,
                observation,
                self.action_manual,
                self.ctrl_nominal,
                self.ctrl_benchmarking,
                self.ctrl_mode,
            )

            self.sys.receive_action(action)
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
    cost_calculator_instance, tol=1e-2
):
    ref_observation_sequences = np.array([[[ 2.        ,  2.        ,  0.8       ],
        [ 4.9880998 ,  4.9880998 , -2.21601201],
        [ 4.97719353,  4.97360993, -2.06669008],
        [ 4.96810111,  4.9568027 , -1.91096986],
        [ 4.96151765,  4.93820179, -1.75170995],
        [ 4.95792054,  4.91853615, -1.59183396],
        [ 4.95750223,  4.89865529, -1.43427677],
        [ 4.96014433,  4.87942237, -1.28193074],
        [ 4.96543868,  4.86160695, -1.13759253],
        [ 4.97274966,  4.84579957, -1.00391181],
        [ 4.98130268,  4.83236372, -0.88334261],
        [ 4.99027923,  4.8214309 , -0.77809826],
        [ 4.99890087,  4.81293423, -0.69011077],
        [ 5.00649042,  4.80666888, -0.62099536],
        [ 5.0125069 ,  4.80236461, -0.5720208 ],
        [ 5.01655783,  4.79975653, -0.54408615],
        [ 5.01839714,  4.79864376, -0.53770421],
        [ 5.01791796,  4.7989295 , -0.55299212],
        [ 5.01514784,  4.80063929, -0.58966926],
        [ 5.01025009,  4.80391627, -0.64706232],
        [ 5.00353023,  4.80899366, -0.72411771],
        [ 4.9954422 ,  4.81614654, -0.81942091],
        [ 4.98658568,  4.82562719, -0.93122239],
        [ 4.97768521,  4.83759208, -1.05746977],
        [ 4.9695446 ,  4.85203259, -1.19584549],
        [ 4.96297543,  4.86872383, -1.34380933],
        [ 4.9587067 ,  4.88720582, -1.49864507],
        [ 4.95729005,  4.90680614, -1.65751032],
        [ 4.95901987,  4.92670471, -1.81748874],
        [ 4.96388652,  4.94603048, -1.97564354]],

       [[ 2.        ,  2.        ,  0.8       ],
        [ 4.98585786,  4.98585786, -2.27077076],
        [ 4.97122692,  4.9684865 , -2.17449957],
        [ 4.956794  ,  4.94755669, -2.06738093],
        [ 4.94338949,  4.92281946, -1.94941483],
        [ 4.93198712,  4.89415674, -1.82060127],
        [ 4.92369075,  4.86163908, -1.68094025],
        [ 4.91970378,  4.82558768, -1.53043178],
        [ 4.92127688,  4.78663639, -1.36907585],
        [ 4.92963068,  4.74578691, -1.19687246],
        [ 4.94585118,  4.70444858, -1.01382161],
        [ 4.97075907,  4.66445153, -0.8199233 ],
        [ 5.00475729,  4.62802077, -0.61517754],
        [ 5.04766712,  4.59769838, -0.39958432],
        [ 5.09856859,  4.57620252, -0.17314364],
        [ 5.15566798,  4.56621613,  0.06414449],
        [ 5.21622116,  4.57010562,  0.31228009],
        [ 5.27654518,  4.58958083,  0.57126314],
        [ 5.33215116,  4.62532169,  0.84109365],
        [ 5.37802558,  4.67661344,  1.12177161],
        [ 5.40907385,  4.74104861,  1.41329704],
        [ 5.42071789,  4.81436703,  1.71566992],
        [ 5.40960894,  4.89051008,  2.02889026],
        [ 5.37437971,  4.96195781,  2.35295805],
        [ 5.3163221 ,  5.02039244,  2.68787331],
        [ 5.23984596,  5.05768607,  3.03363602],
        [ 5.15256048,  5.0671459 ,  3.39024619],
        [ 5.06483562,  5.04487184,  3.75770381],
        [ 4.98875553,  4.991003  ,  4.1360089 ],
        [ 4.93647321,  4.91056951,  4.52516144]],

       [[ 2.        ,  2.        ,  0.8       ],
        [ 4.98585786,  4.98585786, -1.79546459],
        [ 4.9814022 ,  4.9663605 , -1.19817687],
        [ 4.98868333,  4.94773296, -0.57529597],
        [ 5.00546396,  4.9368513 ,  0.06174366],
        [ 5.02542585,  4.93808539,  0.70124765],
        [ 5.0407066 ,  4.95098882,  1.33147638],
        [ 5.04544745,  4.97041881,  1.94086051],
        [ 5.03821394,  4.98906489,  2.51821336],
        [ 5.02197573,  5.00074053,  3.05293624],
        [ 5.00205428,  5.00251134,  3.53521305],
        [ 4.98358375,  4.99484065,  3.95619045],
        [ 4.96986053,  4.98029166,  4.30814041],
        [ 4.96199397,  4.96190369,  4.58460205],
        [ 4.95944518,  4.94206677,  4.78050026],
        [ 4.96080635,  4.92211314,  4.89223887],
        [ 4.96438399,  4.90243573,  4.91776665],
        [ 4.96846273,  4.88285605,  4.85661497],
        [ 4.97133726,  4.8630637 ,  4.70990643],
        [ 4.97128761,  4.84306376,  4.4803342 ],
        [ 4.96668805,  4.82359984,  4.17211262],
        [ 4.95640059,  4.80644851,  3.79089984],
        [ 4.94047054,  4.79435582,  3.34369392],
        [ 4.9208776 ,  4.79034125,  2.83870438],
        [ 4.90178802,  4.79630682,  2.2852015 ],
        [ 4.88868465,  4.81141648,  1.69334613],
        [ 4.88623979,  4.83126648,  1.0740032 ],
        [ 4.89577196,  4.84884879,  0.4385422 ],
        [ 4.9138794 ,  4.85734119, -0.20137148],
        [ 4.93347526,  4.85334093, -0.8339907 ]]])
    first_observation = np.array([ 5.        ,  5.        , -2.35619449])
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
    print(integral_metric)
    #print(abs(integral_metric - 1.318) / 1.318 )
    return abs(3.204 - integral_metric) / 3.204


def final_grade(g1, g2):
    final_grade = min((g1 == 1) * 60 + ceil(max((1 - g2),0) * 40), 100)
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
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


description = "Agent-environment preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

parser = argparse.ArgumentParser(description=description)

parser.add_argument(
    "--ctrl_mode",
    metavar="ctrl_mode",
    type=str,
    choices=["manual", "nominal", "MPC", "RQL", "SQL", "JACS"],
    default="MPC",
    help="Control mode. Currently available: "
    + "----manual: manual constant control specified by action_manual; "
    + "----nominal: nominal controller, usually used to benchmark optimal controllers;"
    + "----MPC:model-predictive control; "
    + "----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; "
    + "----SQL: stacked Q-learning; "
    + "----JACS: joint actor-critic (stabilizing), system-specific, needs proper setup.",
)
parser.add_argument(
    "--dt", type=float, metavar="dt", default=0.01, help="Controller sampling time."
)
parser.add_argument(
    "--t1", type=float, metavar="t1", default=10.0, help="Final time of episode."
)
parser.add_argument(
    "--Nruns",
    type=int,
    default=1,
    help="Number of episodes. Learned parameters are not reset after an episode.",
)
parser.add_argument(
    "--state_init",
    type=str,
    nargs="+",
    metavar="state_init",
    default=["5", "5", "-3*pi/4"],
    help="Initial state (as sequence of numbers); "
    + "dimension is environment-specific!",
)
parser.add_argument(
    "--is_log_data",
    type=str2bool,
    default=False,
    help="Flag to log data into a data file. Data are stored in simdata folder.",
)
parser.add_argument(
    "--is_visualization",
    type=str2bool,
    nargs="?",
    const=True,
    default=True,
    help="Flag to produce graphical output.",
)
parser.add_argument(
    "--is_print_sim_step",
    type=str2bool,
    default=True,
    help="Flag to print simulation data into terminal.",
)
parser.add_argument(
    "--prob_noise_pow",
    type=float,
    default=False,
    help="Power of probing (exploration) noise.",
)
parser.add_argument(
    "--action_manual",
    type=float,
    default=[-5, -3],
    nargs="+",
    help="Manual control action to be fed constant, system-specific!",
)
parser.add_argument(
    "--Nactor",
    type=int,
    default=3,
    help="Horizon length (in steps) for predictive controllers.",
)
parser.add_argument(
    "--pred_step_size_multiplier",
    type=float,
    default=1.0,
    help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.",
)
parser.add_argument(
    "--stage_obj_struct",
    type=str,
    default="quadratic",
    choices=["quadratic", "biquadratic"],
    help="Structure of stage objective function.",
)
parser.add_argument(
    "--R1_diag",
    type=float,
    nargs="+",
    default=[1, 10, 1, 0, 0],
    help="Parameter of stage objective function. Must have proper dimension. "
    + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
)
parser.add_argument(
    "--R2_diag",
    type=float,
    nargs="+",
    default=[1, 10, 1, 0, 0],
    help="Parameter of stage objective function . Must have proper dimension. "
    + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
    + "where diag() is transformation of a vector to a diagonal matrix.",
)
parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
parser.add_argument(
    "--actor_struct",
    type=str,
    default="quad-nomix",
    choices=["quad-lin", "quadratic", "quad-nomix"],
    help="Feature structure (actor). Currently available: "
    + "----quad-lin: quadratic-linear; "
    + "----quadratic: quadratic; "
    + "----quad-nomix: quadratic, no mixed terms.",
)


args = parser.parse_args([])

if not isinstance(args.state_init[0], int):
    for k in range(len(args.state_init)):
        args.state_init[k] = eval(args.state_init[k].replace("pi", str(np.pi)))


def update_globals_for_test(args=args):
    print("kek")
    args.state_init = np.array([2, 2, 0.8])
    args.action_manual = np.array(args.action_manual)
    dim_state = 3
    args.dim_state = dim_state
    args.state_init = np.array(args.state_init)
    args.action_manual = np.array(args.action_manual)
    args.dim_state = 3
    args.dim_input = 2
    args.dim_output = args.dim_state
    args.dim_disturb = 0

    args.dim_R1 = args.dim_output + args.dim_input
    args.dim_R2 = args.dim_R1
    args.pred_step_size = args.dt * args.pred_step_size_multiplier

    args.R1 = np.diag(np.array(args.R1_diag))
    args.R2 = np.diag(np.array(args.R2_diag))
    return args