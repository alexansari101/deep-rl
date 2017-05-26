# deep-rl
## Deep reinforcement learning modules

This project includes modules to implement modern deep reinforcement learning agents and environments.

## Installing
Use python3. I suggest making a virtualenv
Install tensorflow, matplotlib, scipy, PILLOW, (any others it complains about)

    virtualenv -p python3 tensorflow
    source tensorflow/bin/activate
    pip install tensorflow matplotlib scipy PILLOW

Clone this repo, enter directory, run a trial
  
    git clone git@github.com:alexansari101/deep-rl.git
    cd deep-rl
    python ./run_hA3C.py --train
    
This will create a new folder in `./trials/` with tensorboard data, evaluation videos, and checkpoints.
View on tensorboard using the following command, and in a browser window going to `127.0.1.1:6006`
    
    tensorboard --logdir .
    

## Changing environments, agents, rewards, etc

### Environments
You can change the environment through input options. 

    python ./run_hA3C.py --train --env Search

Create new environments in the `environments` directory, and add them to `env_factory`.

Each run creates a new folder for the trial data. For quick testing, run with the `--tmp` option to put files in a `tmp` directory which will be overwritten.


### Agents
Edit `run_hA3C.py` to change the agents. Several different hierarchical agent structures exist in `agents/H_Workers.py`. 
The networks are also stored in the `agents` directory. Each single-level agent consist of a "network" which defines the deep NN, and a "worker" that implements the ac_agent_base class. Note: These networks and workers do some special things to work both hierarchically and asychronously.

### Intrinsic subgoals
The subgoals defining the meta_action structure, and sub_agent intrinsic rewards are in the `intrinsics` folder.

## Overview of how it all works

The environments follow the OpenAI gym structure, taking in actions and returning observations and rewards. An agent takes in these observations and (using its policy) selects actions to perform on the environment. A learning agent will change performance over time, attempting to maximize reward.

In this project, the agent's policy is a DeepNN, and learning occurs using the actor-critic model. Multiple agents learn asynchronously, updating, and pulling from, a shared master network. Agents are also hierarchical. We implement the hierarchy as follows:

A low level agent, `agent_1`, selects actions directly on the environment. Rather than receiving the *extrinsic* reward, it receives an *intrinsic* reward, as defined in `./intrinsics/`, which may be based on the extrinsic reward, meta_actions, state, etc. When selecting and action, `agent_1` uses both the observation from the environment, as well as a meta_action. `agent_1` also exposes an environment interface, defined in `./environments/h_env_wrapper.py`

A high level agent, `agent_0`, selects actions on this wrapped environment. The action space is determined by the `./intrinsics/` used. The reward may be augmented by the intrinsics, but for now we try and keep it as the raw reward.

