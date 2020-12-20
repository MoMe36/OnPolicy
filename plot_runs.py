import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import glob 
import numpy as np 
from scipy.ndimage.filters import gaussian_filter1d as gf1d
plt.style.use('ggplot')

if __name__ == "__main__": 

    smoother = lambda x:  gf1d(x, sigma = 15)
    
    files = glob.glob('./runs/*csv')
    ppo_files = [f for f in files if 'ppo' in f]
    ppg_files = [f for f in files if 'ppg' in f]

    ppo_df = pd.concat([pd.read_csv(f) for f in ppo_files], axis = 1)
    ppg_df = pd.concat([pd.read_csv(f) for f in ppg_files], axis = 1)

    ppo_df['min_val'] = ppo_df.min(axis = 1)
    ppo_df['max_val'] = ppo_df.max(axis = 1)
    ppo_df['mean_val'] = ppo_df.mean(axis = 1)

    ppg_df['min_val'] = ppg_df.min(axis = 1)
    ppg_df['max_val'] = ppg_df.max(axis = 1)
    ppg_df['mean_val'] = ppg_df.mean(axis = 1)

    f, ax = plt.subplots(figsize = (20,14))

    x_ticks= np.arange(0, ppo_df.shape[0])
    plt.fill_between(x_ticks, smoother(ppo_df['min_val']), smoother(ppo_df['max_val']), alpha = 0.2)
    plt.plot(x_ticks, smoother(ppo_df['mean_val']), label = 'PPO')

    plt.fill_between(x_ticks, smoother(ppg_df['min_val']), smoother(ppg_df['max_val']), alpha = 0.2)
    plt.plot(x_ticks, smoother(ppg_df['mean_val']), label = 'PPG')

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('PPO vs PPG')
    plt.legend()

    plt.savefig('./runs/ppo_vs_ppg.png')
    plt.show()