import numpy as np 

rewards = np.zeros((10))
rewards[-1] = 1.
r = 0. 
discounted = np.zeros_like(rewards)
for i in reversed(range(len(rewards))): 
    r = 0.99 * r + rewards[i]
    discounted[i] = r

print(discounted) 
