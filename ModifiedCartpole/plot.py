import numpy as np
import matplotlib.pyplot as plt
def smooth(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        smoothed_data.append(sum(data[start_index:end_index]) / (end_index - start_index))
    return np.array(smoothed_data)

def read_fn(m,r,n):
    Rie=[]
    nonRie=[]
    for i in range(5):
        temp_file_r=np.load(f"./result/Riemannian/Cartpole_seed_{i}_m_{m}_r_{r}_n_{n}_reward.npy")
        temp_file_nr=np.load(f"./result/nonRiemannian/Cartpole_seed_{i}_m_{m}_r_{r}_n_{n}_reward.npy")
        Rie.append(temp_file_r)
        nonRie.append(temp_file_nr)
    Rie=np.array(Rie)
    nonRie=np.array(nonRie)
    # print(np.mean(Rie,0).shape)
    return np.mean(Rie,0),np.std(Rie,0),np.mean(nonRie,0),np.std(Rie,0)
m=4
n=4
r=4
Rie_mean,Rie_std,nonRie_mean,nonRie_std=read_fn(m,r,n)
t=list(range(Rie_mean.shape[0]))
plt.plot(t,smooth(nonRie_mean,10),label='nonRiemannian')
plt.plot(t,smooth(Rie_mean,10),label='Riemannian')
plt.fill_between(t,smooth(nonRie_mean,10)+smooth(nonRie_std,10),smooth(nonRie_mean,10)-smooth(nonRie_std,10),alpha=0.5)
plt.fill_between(t,smooth(Rie_mean,10)+smooth(Rie_std,10),smooth(Rie_mean,10)-smooth(Rie_std,10),alpha=0.5)
plt.legend()
plt.title(f"m_{m}_r_{r}_n_{n}")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig(f"CartPole_m_{m}_r_{r}_n_{n}.png",dpi=400)