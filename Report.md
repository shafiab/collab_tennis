## Introduction

[MADDPG](https://arxiv.org/pdf/1706.02275.pdf) Multi-Agent Deep Deterministic Policy Gradient is an extension of the [DDPG](https://arxiv.org/pdf/1509.02971.pdf) algorithm in a multi-agent setting. The main aspects of MADDPG is following:

- Each agent has its own actor and critic network
- Training is centralized, where agents can share their experience using a shared experience replay buffer
- Actor network for each agent has only access to local information
- Critic network for each agent can be augmented with information about policies of other agent
- Since, we only need the actors during execution, it is decentralized, whereas during training the critic network can use extra information about other agents due to centralized training.

The overall flow of the algorithm is reproduced here from the paper:

[image_1]: alg_flow.png "MADDPG Algorithm"
![Trained Agents][image_1]

## MADDPG Implementation and Issues
I faced a number of challenges while training this model:

- I tried a number of different hyperparameter configurations (more on this in the next section). However, no matter the configurations, the average reward was always zero for the first 1000 episodes that I tried. Then once I increased σ - which represents the variation or the size of the noise in the Ornstein-Uhlenbeck Noise Generation, I started observing positive rewards values.
- However, even though I started observing positive reward after increasing  σ, the average reward was oscillating and ended up going down to zero over time. At this point, my intuition was even though large σ that I am using is good for exploration at the beginning of the training and and resulting in positive rewards, in the middle of the training it would be better to increase exploitation more. So, I decided to add additional paramaeter ε to scale down the noise over time. After adding and playing with this parameter a bit, I started seeing improvement.
- Each agent has a state vector of length 24 and action vector of length 2. Initially, I was using both agents' states (length 48=24+24) and actions (length 4=2+2) as input to the critic network for each agent. However, this was without success and the model was not converging. So, I changed it back so that critic of each agent was only using its own states (length 24) and own actions (length 2) as input.
- Initially, the replay buffers of each agent was only containing the (S,A, R, S) experience tuples for that agent. After, reading the benchmark implementation section of the Udacity project, I realized agents can benefit from sharing experiences. And hence, I started using a shared experience buffer. For ease of implementation, each agent has its own buffer, however the buffer contained experience tuples from both agents.
- I tried experimenting with prioritized experience replay without success. In the end, I ended up using a regular experience replay impementation with shared experience tuples from both agents without priority.

## Model Hyperparameters
### Actor Network
### Critic Network
### Target actor and critic Networks
- delayed copy of actor and critic network, with soft-update parameter tau set to 1e-3

### Experience Replay
- input buffer size of 1e6
- sample/batch size of 512

### Q value
- discount factor gamma was set to 0.9

### Ornstein-Uhlenbeck Noise Generation
- code is the same from the udacity implementation
-  mu=0, theta=0.15, sigma=0.1 was used (in the paper sigma=0.2)

## Reward Plot and Convergence
A reward vs episode plot is presented below. The model reached the target reward in 2154 episodes.

[image_2]: reward_plt.png "Rewards vs. Episodes"
![Trained Agents][image_2]

## Saved Model
- Saved actor weights for the agents are [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_actor_0.pth) and [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_actor_1.pth)
- Saved critic weights for the agents are [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_critic_0.pth) and [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_critic_1.pth)
- The loads can be loaded by following the code in the [notebook](https://github.com/shafiab/collab_tennis/blob/master/Tennis_final.ipynb)
- Since I trained the model using gpu on udacity workspace and then loaded the weight on my macbook to see the trained model at work, I faced an error. Some stack overflow search suggested to include `map_location={'cuda:0': 'cpu'}` while loading the models on cpu and it woked for me.
```
    agent.actor.load_state_dict(torch.load('checkpoint_actor_final.pth', map_location={'cuda:0': 'cpu'}))
    agent.critic.load_state_dict(torch.load('checkpoint_critic_final.pth', map_location={'cuda:0': 'cpu'}))
```
## Future Work
Few future ideas and improvement can be done here:

