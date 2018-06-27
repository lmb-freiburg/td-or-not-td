import sys
import argparse
from os.path import join as p_join
from td_or_not_td.env.env_doom import tasks
from td_or_not_td.env.env_doom import Environment


def get_argparse_parameter():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-main_path', type=str, default=None, help='data output path')
    parser.add_argument('-eval', action='store_true',
                        help='Evaluate main_path snapshots instead of training.')
    parser.add_argument('-eval_one', action='store_true',
                        help='Evaluate one main_path snapshot instead of training, '
                             'specified by eval_one_id value.')
    parser.add_argument('-eval_one_id', type=int, default=24, help=' ')
    parser.add_argument('-show_screen', action='store_true', help=' ')

    parser.add_argument('-game', type=str, default="doom", choices=("doom",), help=' ')
    parser.add_argument('-doom_lvl', type=str, default="battle", choices=tasks, help=' ')
    parser.add_argument('-algorithm', type=str, default="Qmc", choices=("Q", "a3c", "Qmc"),
                        help=' ')

    parser.add_argument('-training_time', type=int, default=int(6e7), help=' ')
    parser.add_argument('-exploration_time', type=int, default=int(5e7), help=' ')
    parser.add_argument('-rms_learning_rate', type=float, default=7.0e-4, help=' ')

    parser.add_argument('-mc_rollout', type=int, default=32, help='Qmc maximal finite horizon')
    parser.add_argument('-td_rollout', type=int, default=20, help='Q, and a3c n-step rollout')

    parser.add_argument('-batch_size', type=int, default=20, help=' ')

    args = parser.parse_args()

    if args.main_path is not None:
        if not (args.eval or args.eval_one):
            save_para_file_path = p_join(args.main_path, "argparse_parameter.txt")
            with open(save_para_file_path, 'w') as file:
                file.write('\n'.join(sys.argv[1:]))
    return args


class Parameter:
    def __init__(self, args=None):
        """Creates all algorithm parameters.

        Args:
            args (Namespace): output of get_argparse_parameter()
        """

        if args is None:
            args = get_argparse_parameter()

        self.main_path = args.main_path

        self.eval_run = args.eval or args.eval_one

        self.single_eval = args.eval_one
        self.single_eval_id = args.eval_one_id

        self.game = args.game
        self.doom_lvl = args.doom_lvl

        self.action_repetition = 4
        self.screen_res_x = 84
        self.screen_res_y = 84
        self.input_image_number = 1

        self.algorithm = args.algorithm

        if self.algorithm == "Qmc":
            self.discount = 1.0
        else:
            self.discount = 0.99

        self.batch_size = args.batch_size
        if self.algorithm in ("Q", "a3c"):
            self.rollout = args.td_rollout
            assert self.batch_size >= self.rollout
            assert self.batch_size % self.rollout == 0

            # Examples of target_v for n-step TD batch:
            #
            # for batch_size=10 and rollout=10 (for simplicity discount=1 and no terminal states):
            # v_offset_list = [0]
            #
            # target_v(S_0) = r_0 + r_1 + r_2 + r_3 + r_4 + r_5 + r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_1) =       r_1 + r_2 + r_3 + r_4 + r_5 + r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_2) =             r_2 + r_3 + r_4 + r_5 + r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_3) =                   r_3 + r_4 + r_5 + r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_4) =                         r_4 + r_5 + r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_5) =                               r_5 + r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_6) =                                     r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_7) =                                           r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_8) =                                                 r_8 + r_9 + V(S_10)
            # target_v(S_9) =                                                       r_9 + V(S_10)
            #
            #
            # for batch_size=10 and rollout=5 (for simplicity discount=1 and no terminal states):
            # v_offset_list = [0, 5]
            #
            # target_v(S_0) =                               r_5 + r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_1) =                                     r_6 + r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_2) =                                           r_7 + r_8 + r_9 + V(S_10)
            # target_v(S_3) =                                                 r_8 + r_9 + V(S_10)
            # target_v(S_4) =                                                       r_9 + V(S_10)
            # target_v(S_5) = r_0 + r_1 + r_2 + r_3 + r_4 + V(S_5)
            # target_v(S_6) =       r_1 + r_2 + r_3 + r_4 + V(S_5)
            # target_v(S_7) =             r_2 + r_3 + r_4 + V(S_5)
            # target_v(S_8) =                   r_3 + r_4 + V(S_5)
            # target_v(S_9) =                         r_4 + V(S_5)

            self.v_offset_list = [i for i in range(0, self.batch_size, self.rollout)]
            self.batch_offset = 0
        elif self.algorithm == "Qmc":
            self.rollout = args.mc_rollout
            self.batch_offset = self.rollout - 1
        else:
            raise ValueError("algorithm unknown")

        if self.algorithm == "Qmc":
            assert self.rollout
            self.prediction_steps = []  # list of finite prediction horizons for the Qmc heads
            i = 0
            while 2 ** i < self.rollout:
                self.prediction_steps.append(2 ** i)
                i += 1
            if self.prediction_steps[-1] != self.rollout:
                self.prediction_steps.append(self.rollout)

            self.number_of_predictions = len(self.prediction_steps)
            assert self.number_of_predictions >= 3
            self.prediction_steps_usage = [0.] * self.number_of_predictions
            self.prediction_steps_usage[-1] = 1.
            self.prediction_steps_usage[-2] = 0.5
            self.prediction_steps_usage[-3] = 0.5

        if self.algorithm in ("Q", "a3c"):
            self.target_v_size = len(self.v_offset_list)
        elif self.algorithm == "Qmc":
            self.target_v_size = self.number_of_predictions

        if self.algorithm == "a3c":
            self.use_eps_exploration = False
        else:
            self.use_eps_exploration = True

        self.exploration_time = args.exploration_time

        self.number_of_agents = 16
        self.number_of_eval_agents = 2
        self.stop_after = args.training_time + 100000

        self.rms_learning_rate = args.rms_learning_rate

        self.replay_memory_size = 1000

        self.run_on_gpu = False
        self.gpu_memory_fraction = 1.0

        self.print_log_occurrence = 10000
        self.save_network_occurrence = 2500000
        self.update_target_occurrence = 10000

        self.number_of_eval_runs = 500

        if self.game == "doom":
            self.use_screen = True
            self.use_vector_input = True

            _, _, self.input_vector_size, _, self.number_of_actions, _ = \
                Environment.get_task_parameter(args.doom_lvl)

        else:
            raise ValueError("unknown game")

        self.qmc_no_target_available_encoding = -10.

        if self.run_on_gpu:
            self.data_format = "NCHW"
        else:
            self.data_format = "NHWC"

        if args.show_screen:
            self.window_visible = True
            self.number_of_agents = 1
            self.number_of_eval_agents = 1
        else:
            self.window_visible = False
