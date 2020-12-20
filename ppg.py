import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import gym 
from torch.distributions import Normal, kl 
from torch.utils.data import Dataset, DataLoader 
from collections import deque 
import pandas as pd 
import matplotlib.pyplot as plt 
from argparse import ArgumentParser 
from tqdm import tqdm

env = gym.make('Pendulum-v0')
obs_size = env.observation_space.shape[0]
ac_size = env.action_space.shape[0]
max_action = env.action_space.high[0]


def discount_rewards(rewards, discount = 0.99): 
    discounted = np.zeros_like(rewards)
    r = 0.
    for i in reversed(range(len(rewards))): 
        discounted[i] = discount * r + rewards[i]
        r = discounted[i]

    return discounted

class XPDataset(Dataset): 

    def __init__(self, states, rewards, actions, next_states): 
        super().__init__()
        self.s = states 
        self.r = rewards
        self.a = actions 
        self.ns = next_states

    def __len__(self): 
        return len(self.r)
    def __getitem__(self, idx): 

        s = torch.tensor(self.s[idx]).float().reshape(-1)
        a = torch.tensor(self.a[idx]).float().reshape(-1)
        r = torch.tensor(self.r[idx]).float().reshape(-1)
        ns = torch.tensor(self.ns[idx]).float().reshape(-1)

        return s,a,r,ns

class RewardDataset(Dataset): 
    def __init__(self, current_buffer): 
        super().__init__()
        self.current_buffer = current_buffer
    def __len__(self): 
        return len(self.current_buffer)
    def __getitem__(self, idx): 
        state, value = self.current_buffer[idx]
        return torch.tensor(state).float().reshape(-1), torch.tensor(value).float().reshape(-1)

get_loader = lambda x : DataLoader(x, batch_size = 64, shuffle = True)
memory_buffer = deque(maxlen = 5000)

class Policy(nn.Module): 

    def __init__(self, obs, ac, hidden = 64): 
        super().__init__()

        self.l1 = nn.Sequential(nn.Linear(obs, hidden), 
                                nn.Tanh(), 
                                nn.Linear(hidden,hidden), 
                                nn.Tanh())
        self.mean_head = nn.Linear(hidden, ac)
        # self.log_std_head = nn.Linear(hidden, ac)
        self.log_std = nn.Parameter(torch.zeros(ac))
        self.value_aux = nn.Linear(hidden, 1)

    def forward(self, state): 

        l1 = self.l1(state)
        mean = self.mean_head(l1)
        # std = self.log_std_head(l1).clamp(-20,2).exp()
        return mean, self.log_std.exp()

    def get_values(self, state): 
        return self.value_aux(self.l1(state))

policy = Policy(obs_size, ac_size)

old_policy = Policy(obs_size, ac_size)
old_policy.load_state_dict(policy.state_dict())

aux_policy = Policy(obs_size, ac_size)
aux_policy.load_state_dict(policy.state_dict())


epsilon_clip = 0.2

value = nn.Sequential(nn.Linear(obs_size, 64), 
                      nn.ReLU(), 
                      nn.Linear(64, 64), 
                      nn.ReLU(), 
                      nn.Linear(64,1))

adam_v = optim.Adam(value.parameters(), lr = 3e-3, weight_decay = 1e-2)
adam_p = optim.Adam(policy.parameters(), lr = 3e-4)


def train(current_epoch, loader, aux_loader,
          epochs_normal = 1, epochs_aux = 6, 
          beta_clone = 1., aux_every = 12): 

    old_policy.load_state_dict(policy.state_dict())
    for epoch in range(epochs_normal): 
        for counter, data in enumerate(loader): 
            s, a, r, ns = data 
            
            state_estimates = value(s)
            value_loss = F.mse_loss(state_estimates, r)

            next_state_estimates = value(ns)
            advantage = r + 0.99 * next_state_estimates - state_estimates
            
            adam_v.zero_grad()
            value_loss.backward()
            adam_v.step()

            mean, std = policy(s)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(a)

            mean_o, std_o = old_policy(s)
            old_dist = Normal(mean_o, std_o)
            old_log_probs = old_dist.log_prob(a)

            policies_ratio = (log_probs - old_log_probs).exp()
            policy_loss = -torch.min(policies_ratio * advantage.detach(), policies_ratio.clamp(1.-epsilon_clip, 1.+epsilon_clip) * advantage.detach()).mean()

            # policy_loss = -(log_probs * advantage.detach()).mean()
            adam_p.zero_grad()
            policy_loss.backward()
            adam_p.step()

    if current_epoch % aux_every == 0: 
        aux_policy.load_state_dict(policy.state_dict())
        for epoch in range(epochs_aux): 
            batch_counter = 0
            for data in aux_loader: 
                s, r = data 
                with torch.no_grad(): 
                    aux_old_mean, aux_old_std = aux_policy(s)
                    aux_dist = Normal(aux_old_mean, aux_old_std)
                current_mean, current_std = policy(s)
                current_dist = Normal(current_mean, current_std)
                kl_loss = kl.kl_divergence(aux_dist, current_dist).mean()
                aux_value_loss = F.mse_loss(policy.get_values(s), r)
                aux_loss = aux_value_loss + beta_clone * kl_loss
                
                adam_p.zero_grad()
                aux_loss.backward()
                adam_p.step()

                state_estimates = value(s)
                value_loss = F.mse_loss(state_estimates, r)
                adam_v.zero_grad()
                value_loss.backward()
                adam_v.step()

                batch_counter += 1 
                if batch_counter == 16:
                    break 

    return state_estimates.mean().item(), value_loss.item(), advantage.mean().item()





def train_agent(episodes, seed, out_name): 

    np.random.seed(seed)
    torch.manual_seed(seed)

    latest_rewards = deque(maxlen = 20)
    track_rewards = deque(maxlen=episodes)

    with open('runs/{}_{}.csv'.format(out_name, seed), 'w') as f: 
        f.write('{}_{}\n'.format(out_name, seed))

    pbar = tqdm(total = episodes)

    for episode in range(episodes): 

        s = env.reset()
        done = False 
        states, actions, rewards, next_states = [],[],[],[]
        ep_rewards = 0.
        while not done: 

            with torch.no_grad(): 
                mean, std = policy(torch.tensor(s).float().reshape(1,-1))
                dist = Normal(mean, std) 
                a = dist.sample().numpy().flatten()

            ns, r, done, _ = env.step(a * max_action)
            states.append(s)
            rewards.append(r)
            actions.append(a)
            next_states.append(ns)

            s = ns 
            ep_rewards += r

            track_rewards.append(r)

        rewards = (np.array(rewards) - np.mean(track_rewards))/np.std(track_rewards)
        rewards_to_go = discount_rewards(rewards)
        loader = get_loader(XPDataset(states, rewards_to_go, actions, next_states))
        
        for i in range(len(states)): 
            memory_buffer.append([list(states[i]), rewards_to_go[i]])

        train_data = train(episode, loader, get_loader(RewardDataset(memory_buffer)))
        
        latest_rewards.append(ep_rewards)
        
        with open('runs/{}_{}.csv'.format(out_name, seed), 'a') as f: 
            f.write('{}\n'.format(ep_rewards))

        pbar.update(1)
        if episodes % 10 == 0: 
            pbar.set_description('Mean R{:.2f}'.format( np.mean(latest_rewards)))

    pbar.close()

if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument('--seed', default = 1234, type = int)
    parser.add_argument('--max_eps', default = 2000, type = int)
    parser.add_argument('--out_name', default = 'ppg')
    args = parser.parse_args()

    train_agent(episodes = args.max_eps, 
                seed = args.seed, 
                out_name = args.out_name)


