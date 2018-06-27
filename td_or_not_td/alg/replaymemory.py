from __future__ import print_function, division
from builtins import range
import numpy as np
import copy
from collections import deque
from itertools import islice


class ReplayMemory:
    def __init__(self, parameter):
        """Contains all information that needs to be fed to TensorFlow.

        Args:
            parameter (config.Parameter): Current parameter.
        """
        self._p = parameter

        self._s = deque(maxlen=self._p.replay_memory_size)
        self._v = deque(maxlen=self._p.replay_memory_size)
        self._a = deque(maxlen=self._p.replay_memory_size)
        self._r = deque(maxlen=self._p.replay_memory_size)
        self._t = deque(maxlen=self._p.replay_memory_size)

        if self._p.data_format == "NHWC":
            def data_shape(i):
                return [i, self._p.screen_res_y, self._p.screen_res_x, self._p.input_image_number]
        else:
            def data_shape(i):
                return [i, self._p.input_image_number, self._p.screen_res_y, self._p.screen_res_x]

        if not self._p.use_vector_input:
            self.next_vector = \
                lambda: np.full([1, self._p.input_vector_size], 0.0)
            self.vector_batch = \
                lambda: np.full([self._p.batch_size, self._p.input_vector_size], 0.0)
            self.vector_list_for_v = \
                lambda: np.full([self._p.target_v_size, self._p.input_vector_size], 0.0)

        if not self._p.use_screen:
            self.next_state = \
                lambda: np.full(data_shape(1), 0.0)
            self.state_batch = \
                lambda: np.full(data_shape(self._p.batch_size), 0.0)
            self.state_list_for_v = \
                lambda: np.full(data_shape(self._p.target_v_size), 0.0)

        # The next state information is pre-saved here until the full
        # information (state, vector, action, reward, is_terminal) is available:
        self.next_s = None
        self.next_v = None

        # for calculating target_v_batch for TD algorithms:
        self.v_prediction_list = None

        self._learning_rate = self._p.rms_learning_rate
        self._delta_learning_rate = (self._learning_rate - 1e-8) / float(self._p.stop_after)
        self.t = 0

        # for logging:
        self.last_episode_reward = 0.
        self.current_episode_reward = 0.

    def next_pre_save(self, state, vector=None):
        """Pre-save next observation until full information is available.

        Args:
            state: Image observation.
            vector: Additional measurement observations.
        """
        assert self.next_s is None
        self.next_s = copy.deepcopy(state)
        if vector is not None:
            self.next_v = copy.deepcopy(vector)

    def save(self, action, reward, is_terminal):
        """Save full transition.

        Args:
            action: One-hot action vector.
            reward: Reward received for current action.
            is_terminal: If True the next state will be from a new episode.
        """
        self.current_episode_reward += reward
        assert self.next_s is not None
        self._s.append(self.next_s)
        self.next_s = None
        if self.next_v is not None:
            self._v.append(self.next_v)
        self._a.append(copy.deepcopy(action))
        self._r.append(copy.deepcopy(reward))
        self._t.append(copy.deepcopy(is_terminal))
        if is_terminal:
            self.last_episode_reward = self.current_episode_reward
            self.current_episode_reward = 0.

    def reset(self):
        """Reset all experience."""
        self._s.clear()
        self._v.clear()
        self._a.clear()
        self._r.clear()
        self._t.clear()

    def get_size(self):
        """Get current experience replay size."""
        return len(self._s)

    # All functions used to feed TensorFlow placeholders:

    def learning_rate(self):
        return self._learning_rate - self._delta_learning_rate * float(self.t)

    def next_state(self):
        return [self.next_s]

    def next_vector(self):
        return [self.next_v]

    def state_list_for_v(self):
        assert self._p.v_offset_list[0] == 0
        s_list = [self.next_s]
        for i in range(1, len(self._p.v_offset_list)):
            s_list.append(self._s[-self._p.v_offset_list[i]])
        return s_list

    def vector_list_for_v(self):
        assert self._p.v_offset_list[0] == 0
        m_list = [self.next_v]
        for i in range(1, len(self._p.v_offset_list)):
            m_list.append(self._v[-self._p.v_offset_list[i]])
        return m_list

    def state_batch(self):
        last = self.get_size() - self._p.batch_offset
        return list(islice(self._s, last - self._p.batch_size, last))

    def vector_batch(self):
        last = self.get_size() - self._p.batch_offset
        return list(islice(self._v, last - self._p.batch_size, last))

    def action_batch(self):
        last = self.get_size() - self._p.batch_offset
        return list(islice(self._a, last - self._p.batch_size, last))

    def reward_batch(self):
        last = self.get_size() - self._p.batch_offset
        return list(islice(self._r, last - self._p.batch_size, last))

    def terminal_batch(self):
        last = self.get_size() - self._p.batch_offset
        return list(islice(self._t, last - self._p.batch_size, last))

    def value_target_batch(self):
        if self._p.algorithm == "Qmc":
            def get_r_sum_list(rm_i):
                r_list = []
                for prediction_step in self._p.prediction_steps:
                    i_off = rm_i + prediction_step - 1
                    assert i_off < self.get_size()
                    # if a terminal state occurs, the target is not used:
                    if any(islice(self._t, rm_i, i_off)):
                        r_list.append(self._p.qmc_no_target_available_encoding)
                    else:
                        r_tmp = self._r[i_off]
                        for j in reversed(range(rm_i, i_off)):
                            r_tmp = self._r[j] + self._p.discount * r_tmp
                        r_list.append(r_tmp)
                return r_list

            last = self.get_size() - self._p.batch_offset
            return [get_r_sum_list(i) for i in range(last - self._p.batch_size, last)]

        elif self._p.algorithm in ("Q", "a3c"):
            r_batch = [0.] * self._p.batch_size
            r = 0.
            for i in range(self._p.batch_size):
                if self._t[-(i + 1)]:
                    r = self._r[-(i + 1)]
                else:
                    if i in self._p.v_offset_list:
                        r = self._r[-(i + 1)] + \
                            self._p.discount * self.v_prediction_list[
                                self._p.v_offset_list.index(i)]
                    else:
                        r = self._r[-(i + 1)] + self._p.discount * r

                r_batch[-(i + 1)] = copy.copy(r)
            return r_batch
        else:
            raise ValueError("algorithm unknown")


if __name__ == "__main__":
    # Tests:

    from td_or_not_td.alg.config import get_argparse_parameter, Parameter

    args = get_argparse_parameter()
    args.algorithm = "Qmc"
    para = Parameter(args)
    para.batch_size = 1

    rm = ReplayMemory(para)

    for n in range(para.replay_memory_size):
        rm.next_pre_save(0.)
        rm.save(0., 1., False)

    assert rm.value_target_batch() == [para.prediction_steps] * para.batch_size

    for n in range(para.replay_memory_size):
        rm.next_pre_save(0.)
        rm.save(0., n, False)

    start_i = rm.get_size() - para.rollout
    print(rm.value_target_batch()[0])
    print(([[
        sum(range(start_i, start_i + i))
        for i in para.prediction_steps
    ]] * para.batch_size)[0])
    assert rm.value_target_batch() == [[
        sum(range(start_i, start_i + i))
        for i in para.prediction_steps
    ]] * para.batch_size

    args = get_argparse_parameter()
    args.algorithm = "Q"
    para = Parameter(args)
    para.discount = 1.

    rm = ReplayMemory(para)

    rm.v_prediction_list = [0.]

    for n in range(para.replay_memory_size):
        rm.next_pre_save(0.)
        rm.save(0., 1., False)
    assert rm.value_target_batch() == list(reversed(range(1, para.rollout + 1)))

    for n in range(para.replay_memory_size):
        rm.next_pre_save(0.)
        rm.save(0., n, False)

    start_i = rm.get_size() - para.rollout
    print(rm.value_target_batch())
    print([
        sum(range(rm.get_size() - (i + 1), rm.get_size()))
        for i in reversed(range(para.rollout))
    ])
    assert rm.value_target_batch() == [
        sum(range(rm.get_size() - (i + 1), rm.get_size()))
        for i in reversed(range(para.rollout))
    ]

    print("passed tests")
