# Collaborative Tennis

## Introduction

In this project, I trained a deep reinforcement learning agent using [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) (Multi-Agent Deep Deterministic Policy Gradient) algorithm to play collaborative tennis on Unity ML-agent. Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. This particular setting is known as the [tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. For this particular project, the problem setup is modified by Udacity and the objective is for the two agents is to keep the ball in play by controlling rackets to bounce a ball over a net.

Here is how the trained agent behave:

[image_1]: tennis.gif "Trained Agents"
![Trained Agents][image_1]


## The Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## How to Navigate this Repo
The repo consists of the following files:
- `Tennis_final.ipynb` contains a jupyter notebook with all the codes to train and run the agents.
- `Report.MD` contains description of the algorithm and hyper-parameters and performance of the algorithm.
- `checkpoint_actor_0.pth`, `checkpoint_actor_1.pth` contains the weights of two trained agents' actor neural networks. `checkpoint_critic_0.pth`, `checkpoint_critic_1.pth`  contains the weights of two trained agents' critic neural networks.  See  `Tennis_final.ipynb` for example of how to load these weights.


## Setup / How to Run?

I trained the agent using gpu in a workspace provided by Udacity. However, the workspace does not allow to see the simulator of the environment. So, once the agents are trained, I loaded the trained networks in a Jupyter Notebook in macbook and observed the behavior of the agents in a pre-built unity environment. The steps for the setup is as follows:

- Follow the instruction in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. 
- Once the setup is done, we can activate the environment and run notbook as follows:
```
source activate drlnd
jupyter notebook
```
This will open a notebook session in the browser.
- The pre-build unity environment `Tennis.app.zip` is also included in this repo.
- So, we can just use the `Tennis_final.ipynb` notebook to train and run the agent. All codes are included in that notebook.
