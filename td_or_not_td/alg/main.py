import tensorflow as tf
import numpy as np
import os
import time
import queue
import threading
from io import open
from random import random, randint
from td_or_not_td.alg.graph import Agent, Graph
from os.path import join as p_join
from td_or_not_td.alg.config import Parameter
from td_or_not_td.alg.replaymemory import ReplayMemory
from td_or_not_td.alg.utils import PeriodicEvent, NetworkSaver, TimeMeasurement

p = Parameter()

if p.game == "doom":
    from td_or_not_td.env.env_doom import Environment
else:
    raise ValueError("unknown game")


class Main:
    """Main runner class, Main.__init__() will start training or evaluating"""
    def __init__(self, parameter=p):
        self.p = parameter
        self.time_m = TimeMeasurement()
        self.g = tf.Graph()
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.p.gpu_memory_fraction)

        if self.p.eval_run:
            print("########## EVAL {} ##########".format(self.p.main_path))
        else:
            print("########## RUNNING {} ##########".format(self.p.main_path))

        if not self.p.run_on_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""

            os.environ['MKL_NUM_THREADS'] = "1"
            os.environ['NUMEXPR_NUM_THREADS'] = "1"
            os.environ['OMP_NUM_THREADS'] = "1"

        if self.p.eval_run:
            if self.p.single_eval:
                log_file_name = 'evaluation_log_of_{}.txt'.format(self.p.single_eval_id)
            else:
                log_file_name = 'evaluation_log.txt'

        else:
            log_file_name = 'training_log.txt'

        self.logfile = None
        self.sess = None
        with open(p_join(self.p.main_path, log_file_name), 'w') as self.logfile, \
                tf.Session(graph=self.g, config=tf.ConfigProto(
                    gpu_options=self.gpu_options,
                    inter_op_parallelism_threads=self.p.number_of_agents,
                    intra_op_parallelism_threads=1)) as self.sess:

            self.graph = Graph(self.p)

            self.rm_list = []
            self.agent_list = []
            for rank in range(self.p.number_of_agents):
                self.rm_list.append(ReplayMemory(self.p))
                self.agent_list.append(Agent(rank, self.sess, self.graph, self.rm_list[-1]))

            saver = tf.train.Saver(
                self.graph.main_var,
                max_to_keep=self.p.stop_after // self.p.save_network_occurrence + 1
            )
            self.saver = NetworkSaver(self.sess, saver, self.p.main_path)

            self.sess.run(tf.global_variables_initializer())

            threads = []
            if not self.p.eval_run:
                global_t = queue.Queue()
                global_t.put(0)
                for n in range(self.p.number_of_agents):
                    threads.append(threading.Thread(target=self.run_train, args=(n, global_t)))
            else:
                for n in range(self.p.number_of_eval_agents):
                    threads.append(threading.Thread(target=self.run_eval, args=(n,)))

            for tr in threads:
                tr.start()
            for tr in threads:
                tr.join()

    def _get_eps(self, t):
        if t < self.p.exploration_time:
            return 1. - t * 0.99 / float(self.p.exploration_time)
        else:
            return 0.01

    def _action_vector(self, action_index):
        a = np.zeros([self.p.number_of_actions], dtype=np.int32)
        a[action_index] = 1
        return a

    # **********************************************************************************************

    def run_train(self, rank, global_t):
        """Training code for one thread.

        Args:
            rank (int): Thread rank.
            global_t (queue.Queue): Global time step
        """

        env = Environment(p.doom_lvl, p.data_format, p.window_visible)

        rm = self.rm_list[rank]
        agent = self.agent_list[rank]

        agent.run("sync")

        eps = self._get_eps(0)

        rm.next_pre_save(*env.get_state())

        def env_step():
            if self.p.use_eps_exploration and random() < eps:
                action_id = randint(0, self.p.number_of_actions - 1)
            else:
                action_id = agent.run("action")
            r, _, is_terminal_state = env.step(action_id)
            rm.save(self._action_vector(action_id), r, float(is_terminal_state))
            rm.next_pre_save(*env.get_state())
            return is_terminal_state

        # pre-learning env runs for the experience replay
        while rm.get_size() < self.p.replay_memory_size:
            env_step()

        if rank == 0:
            if self.p.algorithm == "Q":
                agent.run("sync_target")
            self.time_m.start()

            print_log = PeriodicEvent(self.p.print_log_occurrence)
            save_network = PeriodicEvent(self.p.save_network_occurrence,
                                         first_event_at_zero_time=True)
            update_target = PeriodicEvent(self.p.update_target_occurrence)

        t = 1
        while t < self.p.stop_after + 1:
            agent.run("sync")

            # propagate the environment
            is_terminal = False
            t_n = 0

            if self.p.algorithm in ("a3c", "Q"):
                def step_condition():
                    return t_n < self.p.batch_size and not is_terminal
            else:
                def step_condition():
                    return t_n < self.p.batch_size

            while step_condition():
                is_terminal = env_step()
                t_n += 1

            # updating time
            t = global_t.get() + t_n
            global_t.put(t)

            if rank == 0:
                print_log.update_time(t)
                save_network.update_time(t)
                update_target.update_time(t)

            rm.t = t

            if self.p.use_eps_exploration:
                eps = self._get_eps(t)

            # get value prediction for the target calculations in TD
            if self.p.algorithm in ("a3c", "Q"):
                rm.v_prediction_list = agent.run("v_list")

            # print log info
            if rank == 0:
                if print_log.is_event():
                    loss = agent.run("loss")

                    self.time_m.end()
                    time10000 = self.time_m.get()
                    self.time_m.reset()
                    self.time_m.start()

                    self.logfile.write("{}  {}  {}  {}\n".format(
                        int(t % self.p.save_network_occurrence // self.p.print_log_occurrence),
                        time10000,
                        int(time10000 * float(
                            (self.p.stop_after - t) // self.p.print_log_occurrence) / 3600.),
                        rm.last_episode_reward))
                    self.logfile.flush()

                    st = "[{:3}/{:3}]   " \
                         + "[Time:{:5.1f}]   " \
                         + "[Time left:{:5} h]   " \
                         + "[Reward:{:6.1f}]   "

                    for i in range(len(loss)):
                        st += "[Loss:{:10.6f}]   "

                    print(st.format(
                        int(t % self.p.save_network_occurrence // self.p.print_log_occurrence),
                        self.p.save_network_occurrence // self.p.print_log_occurrence,
                        time10000,
                        int(time10000 * float(
                            (self.p.stop_after - t) // self.p.print_log_occurrence) / 3600.),
                        rm.last_episode_reward,
                        *loss), flush=True)

            # train step
            agent.run("train")

            if rank == 0:
                if update_target.is_event():
                    if self.p.algorithm == "Q":
                        agent.run("sync_target")

                if save_network.is_event():
                    self.saver.save()

    # **********************************************************************************************

    @staticmethod
    def eval_print(rank, string, file):
        print("rank {}:  {}".format(rank, string), flush=True)
        file.write(string + "\n")
        file.flush()

    def run_eval(self, rank):
        """Evaluating code for one thread.

        Args:
            rank (int): Thread rank.
        """

        with open(p_join(self.p.main_path, "eval_log_{}.txt".format(rank)), 'w') as status_log_file:
            env = Environment(p.doom_lvl, p.data_format, p.window_visible)

            rm = self.rm_list[rank]
            agent = self.agent_list[rank]

            lock = threading.Lock()

            last_file_id = self.p.stop_after // self.p.save_network_occurrence + 1

            if self.p.single_eval:
                if rank != 0:
                    return
                else:
                    file_id = self.p.single_eval_id
                    last_file_id = self.p.single_eval_id + 1
                    self.eval_print(rank, "single evaluation of {}  ".format(file_id),
                                    status_log_file)
            else:
                file_id = rank

                output_str = "evaluation of:  "
                while file_id < last_file_id:
                    output_str += "{}  ".format(file_id)
                    file_id += self.p.number_of_eval_agents
                self.eval_print(rank, output_str, status_log_file)
                file_id = rank

            while file_id < last_file_id:
                file_path = p_join(self.p.main_path, "model_{}.ckpt".format(file_id))
                found_save = False
                self.eval_print(rank, "waiting for model_{}.ckpt".format(file_id), status_log_file)
                while not found_save:
                    if os.path.isfile(file_path + ".index") and os.path.isfile(file_path + ".meta"):
                        time.sleep(3.)
                        try:
                            self.saver.load(file_id)
                            found_save = True
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except tf.errors.NotFoundError as e:
                            print(str(e))
                            time.sleep(5.)
                    else:
                        time.sleep(30.)
                agent.run("sync")

                self.eval_print(
                    rank,
                    "model imported | beginning evaluation of model_{}.ckpt".format(file_id),
                    status_log_file
                )

                score = np.full([self.p.number_of_eval_runs], 0.)
                score2 = np.full([self.p.number_of_eval_runs], 0.)
                for i in range(self.p.number_of_eval_runs):
                    is_terminal = False
                    while not is_terminal:
                        rm.next_pre_save(*env.get_state())
                        action_id = agent.run("action")
                        reward, reward_eval, is_terminal = env.step(action_id)
                        rm.save(self._action_vector(action_id), reward, float(is_terminal))
                        score[i] += reward
                        score2[i] += reward_eval

                avg_score = sum(score) / self.p.number_of_eval_runs
                avg_score2 = sum(score2) / self.p.number_of_eval_runs

                with lock:
                    self.logfile.write("{}  {}  {}\n".format(file_id, avg_score, avg_score2))
                    self.logfile.flush()

                self.eval_print(rank, "finished evaluation", status_log_file)
                file_id += self.p.number_of_eval_agents

            self.eval_print(rank, "all evaluations are finished", status_log_file)


if __name__ == "__main__":
    Main()
