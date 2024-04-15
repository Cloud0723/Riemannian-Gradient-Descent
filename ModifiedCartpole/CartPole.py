"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

a2c: Agent uses Advantage Actor Critic algorithm

additionally using
https://arxiv.org/pdf/2305.16901.pdf
"""
import gym
from a2c import A2C
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import random
# 创建参数解析器
parser = argparse.ArgumentParser(description='An example script with named arguments.')

# 添加参数
parser.add_argument('--m', type=int)
parser.add_argument('--r', type=int)
parser.add_argument('--n', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--use_riem', type=int)
# LR = .01  # Learning rate

args = parser.parse_args()
# Init actor-critic agent
use_adam = True

use_riem = args.use_riem
M=args.m
N=args.n
R=args.r
SEED=args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
LR = .001
beta = 0.001
MAX_EPISODES = 2000  # Max number of episodes

if use_riem:
    algo_name = "Riemannian"
else:
    algo_name="nonRiemannian"

agent = A2C(gym.make('CartPole-v0'), random_seed=SEED, Riemannian = True, m=M, n=N, r=R)
#orthogonal initialize
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
agent.actor.apply(initialize_weights)
agent.critic.apply(initialize_weights)

# Init optimizers
if use_adam:
    actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
    critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)
else:
    actor_optim = optim.SGD(agent.actor.parameters(), lr=LR, weight_decay=1e-4)
    critic_optim = optim.SGD(agent.critic.parameters(), lr=LR, weight_decay=1e-4)
    
def sym(x: torch.Tensor):  # pragma: no cover
    return 0.5 * (x.transpose(-1, -2) + x)

#Project gradient to tangent space
def proju( x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return u - x @ sym(x.transpose(-1, -2) @ u)

#Project gradient to Stiefel Manifold from tangent space
def projx(x: torch.Tensor) -> torch.Tensor:
    # print(x.shape)
    U, _, V = torch.linalg.svd(x, full_matrices=False)
    # print(torch.matmul(U, V).shape)
    # raise NotImplementedError
    return torch.matmul(U, V)
    # return torch.einsum("...ik,...kj->...ij", U, V)

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

def print_all_params(model):
    for name, param in model.named_parameters():
        if 'rie' in name:
            print(name)
            
            print(param.data[0][0].numpy()) 
            print(param.grad.data[0][0].numpy())
            
def optimizer_proj_tan(model, optim):
    for name, param in model.named_parameters():
        if 'rie' in name:
            param_grad = param.grad.data.detach()
            param_grad = proju(param.data.detach(), param_grad)
            param.grad.data = param_grad
    
def optimizer_retraction(model, optim):
    for name, param in model.named_parameters():
        if 'rie' in name:
            # param_grad = param.grad.data.detach()
            param_ret = projx(param.data.detach())
            param.data = param_ret


r = []  # Array containing total rewards
avg_r = 0  # Value storing average reward over last 100 episodes

loss_critic=[]
loss_agent=[]
batch_size = 1
for episode in range(MAX_EPISODES):
    critic_optim.zero_grad()
    actor_optim.zero_grad()
    # agent_params_before_update = {name : param.clone().detach() for name, param in agent.actor.named_parameters() if 'rie' in name}
    # critic_params_before_update = {name : param.clone().detach() for name, param in agent.critic.named_parameters() if 'rie' in name} 
    
    total_critic_loss = 0
    total_actor_loss = 0
    for batch in range(batch_size):
        rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode(render=False)
        r.append(total_reward)
        l_actor, l_critic = agent.compute_loss(action_p_vals = action_lp_vals, G = rewards, V = critic_vals)
        loss_critic.append(l_critic.clone().detach())
        loss_agent.append(l_actor.clone().detach())
        total_critic_loss += l_critic
        total_actor_loss += l_actor
    total_critic_loss.backward()
    total_actor_loss.backward()

    if use_riem:
        optimizer_proj_tan(agent.critic, critic_optim)

    actor_optim.step()
    critic_optim.step()

    if use_riem:
        optimizer_retraction(agent.critic, critic_optim)

    # np.save(f"./result/{algo_name}/Cartpole_seed_{SEED}_m_{M}_r_{R}_n_{N}_reward.npy",r)

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



