"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

a2c: Agent uses Advantage Actor Critic algorithm

"""
import gym
from a2c import A2C
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# LR = .01  # Learning rate
LR = .01
SEED = None  # Random seed for reproducibility
MAX_EPISODES = 1000  # Max number of episodes

# Init actor-critic agent
agent = A2C(gym.make('CartPole-v0'), random_seed=SEED)
algo_name="Riemannian"
#orthogonal initialize
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
agent.actor.apply(initialize_weights)
agent.critic.apply(initialize_weights)


# Init optimizers
# actor_optim = optim.SGD(agent.actor.parameters(), lr=LR)
# critic_optim = optim.SGD(agent.critic.parameters(), lr=LR)
#Adagrad, Adam can work
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

#
# Train
#

r = []  # Array containing total rewards
avg_r = 0  # Value storing average reward over last 100 episodes
def sym(x: torch.Tensor):  # pragma: no cover
    return 0.5 * (x.transpose(-1, -2) + x)


#Project gradient to tangent space
def proju( x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return u - x @ sym(x.transpose(-1, -2) @ u)

#Project gradient to Stiefel Manifold from tangent space
def projx(x: torch.Tensor) -> torch.Tensor:
    U, _, V = torch.linalg.svd(x, full_matrices=False)
    return torch.einsum("...ik,...kj->...ij", U, V)

def check_point_on_manifold(
        x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) :
        xtx = x.transpose(-1, -2) @ x
        xxt = x @ x.transpose(-1, -2)
        # print(k)
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        xxt[..., torch.arange(x.shape[-2]), torch.arange(x.shape[-2])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol) or torch.allclose(xxt, xxt.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

def check_vector_on_tangent(
        x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None
loss_critic=[]
loss_agent=[]
for episode in range(MAX_EPISODES):
    critic_optim.zero_grad()
    actor_optim.zero_grad()
    rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode(render=False)
    r.append(total_reward)
    agent_params_before_update = [param.clone().detach() for name, param in agent.actor.named_parameters() if 'weight' in name]
    critic_params_before_update = [param.clone().detach() for name, param in agent.critic.named_parameters() if 'weight' in name]
    l_actor, l_critic = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)
    loss_critic.append(l_critic.clone().detach())
    loss_agent.append(l_actor.clone().detach())
    l_actor.backward()
    l_critic.backward()

    actor_optim.step()
    critic_optim.step()

    agent_params_after_update = [param.clone().detach() for name, param in agent.actor.named_parameters() if 'weight' in name]
    critic_params_after_update = [param.clone().detach() for name, param in agent.critic.named_parameters() if 'weight' in name]

    agent_params_diff = [param_after - param_before for param_after, param_before in zip(agent_params_before_update, agent_params_after_update)]
    critic_params_diff = [param_after - param_before for param_after, param_before in zip(critic_params_before_update, critic_params_after_update)]
    #agent update
    agent_tangent_grad = [proju(i,j) for i,j in zip(agent_params_before_update,agent_params_diff)]
    agent_stiefel_param = [projx(i+j) for i,j in zip (agent_params_before_update, agent_tangent_grad)]
    
    # agent.actor.load_state_dict(torch.load("perfect_policy.pt"))
    agent_state_dict = agent.actor.state_dict()
    count=0
    for i, (name, param) in enumerate(agent.actor.named_parameters()):
        if 'weight' in name:
            _, sigma, _=torch.linalg.svd(agent_state_dict[name], full_matrices=False)
            # print(sigma)
            agent_state_dict[name] = agent_stiefel_param[count]
            count+=1
    # if algo_name=="Riemannian":
    #     agent.actor.load_state_dict(agent_state_dict) #comment it for Adam gradient descent for actor

    #critic update
    critic_tangent_grad = [proju(i,j) for i,j in zip(critic_params_before_update,critic_params_diff)]
    critic_stiefel_param = [projx(i+j) for i,j in zip (critic_params_before_update, critic_tangent_grad)]
    critic_state_dict = agent.critic.state_dict()
    count=0
    for i, (name, param) in enumerate(agent.critic.named_parameters()):
        # print(name)
        if 'weight' in name:
            # if '2.weight' in name:
            #     critic_state_dict[name] = critic_stiefel_param[count]
            critic_state_dict[name] = critic_stiefel_param[count]
            # print(check_point_on_manifold(critic_state_dict[name]))
            count+=1

    if algo_name=="Riemannian":
        agent.critic.load_state_dict(critic_state_dict) #comment it for Adam gradient descent for critic
    
    if not algo_name=="Riemannian":
        np.save("A2C_critic_loss.npy",loss_critic)
        np.save("A2C_agent_loss.npy",loss_agent)
        np.save("A2C_cartpole.npy",r)
    else:
        # print("in_it")
        np.save("A2C_critic_loss_Riemannian.npy",loss_critic)
        np.save("A2C_agent_loss_Riemannian.npy",loss_agent)
        np.save("A2C_cartpole_Riemannian.npy",r)
    # Check average reward every 100 episodes, print, and end script if solved
    if len(r) >= 100:  # check average every 100 episodes

        episode_count = episode - (episode % 100)
        prev_episodes = r[len(r) - 100:]
        avg_r = sum(prev_episodes) / len(prev_episodes)
        if len(r) % 100 == 0:
            print(f'Average reward during episodes {episode_count}-{episode_count+100} is {avg_r.item()}')
        if avg_r > 220:
            # print(f"Solved CartPole-v0 with average reward {avg_r.item()}")
            torch.save(agent.actor.state_dict(),'perfect_policy.pt')
            break


#
# Test
#
for _ in range(100):
    agent.test_env_episode(render=False)