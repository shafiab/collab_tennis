## Introduction

[MADDPG](https://arxiv.org/pdf/1706.02275.pdf) Multi-Agent Deep Deterministic Policy Gradient is an extension of the [DDPG](https://arxiv.org/pdf/1509.02971.pdf) algorithm in a multi-agent setting. The main aspects of MADDPG is following:

- Each agent has its own actor and critic network
- Training is centralized, where agents can share their experience using a shared experience replay buffer
- Actor network for each agent has only access to local information
- Critic network for each agent can be augmented with information about policies of other agent
- Since, we only need the actors during execution, it is decentralized, whereas during training the critic network can use extra information about other agents due to centralized training.

The overall flow of the algorithm is reproduced here from the paper:


