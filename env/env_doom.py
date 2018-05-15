from __future__ import print_function, division
from builtins import range

import numpy as np
import time
import random
import os
import inspect
from PIL import Image
from os.path import join as p_join
from itertools import product
from vizdoom import *


class Environment(object):
  def __init__(self, doom_lvl, data_format="NCHW", window_visible=False, human=False):
    """Create doom environment.

    Args:
      doom_lvl (str): Name of the doom task.
      data_format (str): NCHW or NHWC
      window_visible (bool): Shows the screen of the game if True.
      human (bool): test the doom task if True.
    """

    self.action_repetition = 4
    self.screen_res_x = 84
    self.screen_res_y = 84
    self.data_format = data_format
    self.dtype = np.float32
    self.window_visible = window_visible
    self.human = human

    self.game = DoomGame()
    self.sleep_time = 0.02
    self.time = 0

    self.lvl_id = 0  # for multitexture environment

    self.ammo = None
    self.hp = None
    self.frags = None

    self.input_vector_size = None
    self.task_type = None
    self.actions = None
    self.cfg_path = None
    self.number_of_actions = None
    self.manymaps = None
    self._select_task(doom_lvl)

    doom_lvl_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    self.game.load_config(p_join(doom_lvl_path, self.cfg_path))
    self.game.set_screen_resolution(ScreenResolution.RES_160X120)
    self.game.set_screen_format(ScreenFormat.GRAY8)
    self.game.set_window_visible(self.window_visible)

    if human:
      self.game.add_game_args("+freelook 1")
      self.game.set_screen_resolution(ScreenResolution.RES_640X480)
      self.game.set_window_visible(True)
      self.game.set_mode(Mode.SPECTATOR)

    self.game.init()
  
  def __del__(self):
    self.game.close()

  def get_state(self):
    """Get current state. Has to be called after each step"""
    state = self.game.get_state()

    img = Image.fromarray(state.screen_buffer)
    img = img.resize((self.screen_res_y, self.screen_res_x))
    if self.task_type == "battle":
      self.frags = state.game_variables[2]
      self.hp = state.game_variables[1]
      self.ammo = state.game_variables[0]
      measurements = [self.hp / 100., self.ammo / 50., self.time / 2100.]
    elif self.task_type == "health":
      self.hp = state.game_variables[0]
      measurements = [self.hp / 100., self.time / 2100.]
    else:
      raise ValueError("unknown task_type")

    if self.data_format == "NHWC":
      return np.reshape(np.array(img, dtype=self.dtype) / 255.,
                        [self.screen_res_y, self.screen_res_x, 1]), measurements
    elif self.data_format == "NCHW":
      return np.reshape(np.array(img, dtype=self.dtype) / 255.,
                        [1, self.screen_res_y, self.screen_res_x]), measurements
    else:
      raise ValueError("unknown data_format")

  def step(self, action_id):
    """Propagates the environment.

    Args:
      action_id (int): Index of the action.

    Returns:
      float: Reward for training.
      float: Reward for evaluation.
      bool: True if terminal state, False otherwise.
    """
    if not self.human:
      r_train = self.game.make_action(self.actions[action_id], self.action_repetition)
    else:
      r_train = 0
      for _ in range(4):
        self.game.advance_action()
        r_train += self.game.get_last_reward()
        if self.game.is_episode_finished():
          break

    r_eval = 0

    self.time += 1

    if self.window_visible:
      self._wait()

    if self.game.is_episode_finished():
      self.time = 0
      if self.task_type == "battle":
        # Battle performance is evaluated by the amount of frags [Dosovitskiy & Koltun 2017]
        r_eval = self.frags
      elif self.task_type == "health":
        # Health gathering performance is evaluated by the health amount at the end of the episode
        r_eval = self.hp
      else:
        raise ValueError("unknown task_type")

      if self.manymaps:
        self.lvl_id += 1
        self.game.set_doom_map('MAP{:02d}'.format(self.lvl_id % 89 + 1))

      self.game.new_episode()
      
      return r_train, r_eval, True
    else:
      return r_train, r_eval, False
  
  def _wait(self):
    time.sleep(self.sleep_time * self.action_repetition)


  def _select_task(self, doom_lvl):
    """Sets self.manymaps, self.input_vector_size, self.task_type, self.actions, self.cfg_path, and
    self.number_of_actions according to the selected doom_lvl"""

    health_gathering_path_dict = dict(
      navigation="scenarios/navigation.cfg",
      hg_normal="scenarios/const_health_gathering_1.cfg",
      hg_normal_health_reward="scenarios/const_health_gathering_1_r_hp.cfg",
      hg_normal_many_textures="scenarios/const_health_gathering_1_hexen_manymaps_mix.cfg",
      hg_sparse="scenarios/const_health_gathering_2.cfg",
      hg_very_sparse="scenarios/const_health_gathering_3.cfg",
      hg_delay_2="scenarios/const_health_gathering_1_r_delay_2.cfg",
      hg_delay_4="scenarios/const_health_gathering_1_r_delay_4.cfg",
      hg_delay_8="scenarios/const_health_gathering_1_r_delay_8.cfg",
      hg_delay_16="scenarios/const_health_gathering_1_r_delay_16.cfg",
      hg_delay_32="scenarios/const_health_gathering_1_r_delay_32.cfg",
      hg_terminal_health_m_1="scenarios/const_health_gathering_2_m_1.cfg",
      hg_terminal_health_m_2="scenarios/const_health_gathering_2_m_2.cfg",
      hg_terminal_health_m_3="scenarios/const_health_gathering_2_m_3.cfg"
    )
    battle_path_dict = dict(
      battle="scenarios/labyrinth_with_monsters_hexen_r.cfg",
      battle_2="scenarios/labyrinth_with_monsters_difficult_hexen_r.cfg"
    )

    if doom_lvl in health_gathering_path_dict:
      self.task_type = "health"
      self.cfg_path = health_gathering_path_dict[doom_lvl]
      self.input_vector_size = 2
      self.actions = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]]
    elif doom_lvl in battle_path_dict:
      self.task_type = "battle"
      self.cfg_path = battle_path_dict[doom_lvl]
      self.input_vector_size = 3
      self.actions = [list(el) for el in product([0, 1], repeat=8)]
    else:
      raise ValueError("unknown doom_lvl")

    self.number_of_actions = len(self.actions)

    if doom_lvl in ("hg_normal_many_textures",):
      self.manymaps = True
    else:
      self.manymaps = False


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-doom_lvl', type=str, default="hg_normal_many_textures", choices=(
    "navigation", "battle", "battle_2",
    "hg_normal", "hg_sparse", "hg_very_sparse",
    "hg_normal_health_reward", "hg_normal_many_textures",
    "hg_delay_2", "hg_delay_4", "hg_delay_8", "hg_delay_16", "hg_delay_32",
    "hg_terminal_health_m_1", "hg_terminal_health_m_2", "hg_terminal_health_m_3"
  ))
  args = parser.parse_args()

  env = Environment(args.doom_lvl, human=True)

  for _ in range(100):
    terminal = False
    while not terminal:
      env.get_state()
      reward, reward_eval, terminal = env.step(random.randint(0, env.number_of_actions - 1))
      if reward != 0.:
        print("REWARD:", reward)
      if reward_eval != 0.:
        print("EVAL_REWARD:", reward_eval)
    print("RESTARTED")
