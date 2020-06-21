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


## Saved Model
- Saved actor weights for the agents are [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_actor_0.pth) and [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_actor_1.pth)
- Saved critic weights for the agents are [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_critic_0.pth) and [here](https://github.com/shafiab/collab_tennis/blob/master/checkpoint_critic_1.pth)
- The loads can be loaded by following the code in the [notebook](https://github.com/shafiab/collab_tennis/blob/master/Tennis_final.ipynb)
- Since I trained the model using gpu on udacity workspace and then loaded the weight on my macbook to see the trained model at work, I faced an error. Some stack overflow search suggested to include `map_location={'cuda:0': 'cpu'}` while loading the models on cpu and it woked for me.
```
    agent.actor.load_state_dict(torch.load('checkpoint_actor_final.pth', map_location={'cuda:0': 'cpu'}))
    agent.critic.load_state_dict(torch.load('checkpoint_critic_final.pth', map_location={'cuda:0': 'cpu'}))
```

