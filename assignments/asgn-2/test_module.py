import matplotlib.pyplot as plt
import numpy as np
from asgn_2_stub import update_globals_for_test
import pandas as pd
from rcognita.rcognita import visuals
from IPython.display import display, Markdown
from rcognita.rcognita.utilities import upd_line
from rcognita.rcognita.utilities import reset_line
from rcognita.rcognita.utilities import upd_text
import numpy.linalg as la


class Animator3WRobotNI_traj(visuals.Animator3WRobotNI):
    def __init__(self, objects=[], pars=[]):
        super().__init__(objects, pars)
        self.full_trajectory = []

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
    print(abs(integral_metric - 1.318) / 1.318)
    return abs(integral_metric - 1.318) / 1.318


def final_grade(g1, g2):
    final_grade = (g1 == 1) * 60 + int((1 - g2) * 40) + 1
    display(Markdown(f"<strong> Your grade is: {final_grade}</strong>"))
    if final_grade > 95:
        print("Perfect. You passed.")
    elif final_grade > 90:
        print("Good. You passed.")
    if final_grade < 85:
        print("\x1b[31m Minimal value for passing is 85\x1b[0m")
