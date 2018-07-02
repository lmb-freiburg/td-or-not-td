## Code for the paper: TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning

**Artemij Amiranashvili, Alexey Dosovitskiy, Vladlen Koltun and Thomas Brox, ICLR 2018 ([paper link](https://lmb.informatik.uni-freiburg.de/projects/tdornottd/)).**

**Includes the ViZDoom controlled reinforcement learning tasks and asynchronous Qmc, n-step Q, and A3C implementations.**

### Dependencies:
- Python3
- TensorFlow (tested with version 1.0)
- ViZDoom and its dependencies
- pillow

We used ViZDoom version 1.1.1 in our experiments. Since ViZDoom version 1.1.5 uses different textures for health packs, it might slightly affect the performance.

### Installation:

Enter the `td-or-not-td` directory and run:

    pip3 install -e .

### Algorithms:

The three available algorithms are: 

- `Qmc` on-policy asynchronous dueling Monte Carlo Q-learning with a finite horizon
- `Q` on-policy asynchronous dueling n-step DDQN
- `a3c` on-policy asynchronous advantage actor critic

### Running:

##### Training

    python3 td_or_not_td/alg/main.py -main_path PATH_TO_DATA_OUTPUT -algorithm Qmc -doom_lvl hg_normal

##### Evaluation

Evaluate all network snapshots created by the training:

    python3 td_or_not_td/alg/main.py -eval -main_path PATH_TO_DATA_OUTPUT -algorithm Qmc -doom_lvl hg_normal

As long as main_path exists, the evaluation can begin in parallel or before the training. In those cases the script will wait until the network snapshots appear.
The evaluation results are stored in the `evaluation_log.txt` file at main_path. The first column shows the snapshot id, the second the training reward and the third the evaluation reward.

---

To see all available arguments run:

    python3 td_or_not_td/alg/main.py -h

### Tasks:

The tasks are selected by the `doom_lvl` argument. The tasks can be tested by running:

    python3 td_or_not_td/env/env_doom.py [<doom_lvl>]

List of all `doom_lvl` names:

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
- Basic health gathering with multiple textures:
  - `hg_normal_many_textures`

- Additional Tasks:
  - `navigation`
  - `battle`
  - `battle_2`

The Battle tasks are based on the environments from [Learning to Act by Predicting the Future](https://github.com/IntelVCL/DirectFuturePrediction) by Alexey Dosovitskiy and Vladlen Koltun, ICLR 2017.

If you use the new ViZDoom tasks or the algorithm implementations in your research, please cite the following paper:

    @InProceedings{adkb2018td,
        author    = {Amiranashvili, Artemij and Dosovitskiy, Alexey and Koltun, Vladlen and Brox, Thomas},
        title     = {TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning},
        booktitle = "International Conference on Learning Representations (ICLR)",
        year      = {2018},
        url       = "https://lmb.informatik.uni-freiburg.de/projects/tdornottd/"
    }
