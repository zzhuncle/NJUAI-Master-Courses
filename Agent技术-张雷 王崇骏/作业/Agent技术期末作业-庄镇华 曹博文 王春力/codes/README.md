# Codes for the Agent Courses

## Note
The implementation of the following methods can also be found in this codebase, which are finished by the authors of following papers:

- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

The following command train NDQ on the didactic task `matrix_game_2 `.

```shell
python3 src/main.py 
--config=qplex 
--env-config=matrix_game_2 
with 
local_results_path='../../../tmp_DD/sc2_bane_vs_bane/results/' 
save_model=True use_tensorboard=True 
save_model_interval=200000 
t_max=210000 
epsilon_finish=1.0
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To train QPLEX on SC2 online setting tasks, run the following command:

```shell
python3 src/main.py 
--config=qplex_qatten_sc2 
--env-config=sc2 
with 
env_args.map_name=3s5z 
env_args.seed=1 
local_results_path='../../../tmp_DD/sc2_3s5z/results/' 
save_model=True 
use_tensorboard=True 
save_model_interval=200000 
t_max=2100000 
num_circle=2
```

SMAC maps can be found in in the folder `QPLEX_smac_env` of supplymentary material.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 
