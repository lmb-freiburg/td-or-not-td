# ViZDoom controlled reinforcement learning tasks

Code for the ViZDoom tasks of the paper [TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning](https://lmb.informatik.uni-freiburg.de/projects/tdornottd/), Artemij Amiranashvili, Alexey Dosovitskiy, Vladlen Koltun and Thomas Brox, ICLR 2018

## Dependencies:

- ViZDoom and its dependencies
- pillow

We used ViZDoom version 1.1.1 in our experiments. Since ViZDoom version 1.1.5 uses different textures for health packs, it might affect the performance.

## Tasks:

The tasks are selected by the `doom_lvl` argument of the `Environment` class. The tasks can be tested by running 

    python3 env_doom.py [<doom_lvl>]

List of `doom_lvl` names:

- Basic health gathering:
  - `hg_normal`
- Basic health gathering with health-based reward:
  - `hg_normal_health_reward`
- Reward Sparsity:
  - `hg_normal`
  - `hg_sparse`
  - `hg_very_sparse`
- Reward Delay:
  - `hg_normal`
  - `hg_delay_2`
  - `hg_delay_4`
  - `hg_delay_8`
  - `hg_delay_16`
  - `hg_delay_32`
- Terminal health packs:
  - `hg_terminal_health_m_1`
  - `hg_terminal_health_m_2`
  - `hg_terminal_health_m_3`

- Additional Tasks:
  - `navigation`
  - `battle`
  - `battle_2`

The Battle tasks are based on the environments from [Learning to Act by Predicting the Future](https://github.com/IntelVCL/DirectFuturePrediction) by Alexey Dosovitskiy and Vladlen Koltun, ICLR 2017.

If you use the new doom tasks in your research, please cite the following paper:

    @InProceedings{adkb2018td,
    author    = {Artemij Amiranashvili and Alexey Dosovitskiy and Vladlen Koltun and Thomas Brox},
    title     = {TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning},
    booktitle = "International Conference on Learning Representations (ICLR)",
    year      = {2018},
    url       = "https://lmb.informatik.uni-freiburg.de/projects/tdornottd/"
    }
