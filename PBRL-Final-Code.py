
# coding: utf-8

# In[1]:


#Setting up the problem and solving for optimal value function
import mdptoolbox, mdptoolbox.example
import numpy as np
import random
from numpy.random import seed

np.random.seed(100)
random.seed(110)

states = 100
actions = 20
stages = 10

discount= 0.99

episodes = 500
sub_episodes =100

store_values = np.zeros((episodes,1))
epsilon_store_values = np.zeros((episodes,1))
count = np.zeros((states,actions,states))

P, R = mdptoolbox.example.rand(states, actions)
fh = mdptoolbox.mdp.FiniteHorizon(P, R, discount, stages)
fh.run()
optimal_value = fh.V[0][0]

print(optimal_value)

for i in range(episodes):
    for j in range(sub_episodes):
        s = 0
        for k in range(stages):
            a = np.random.randint(actions)
            next_state = np.random.choice(states,1,p = P[a,s,:])[0]
            count[s][a][next_state] += 1
            s = next_state
    
    new_P = np.zeros((actions,states,states))
    
    for a in range(actions):
        for s1 in range(states):
            if np.sum(count[s1,a] >0):
                for s2 in range(states):
                    new_P[a,s1,s2] = count[s1,a,s2]/(np.sum(count[s1,a]))
            else:
                new_P[a,s1,s1] = 1
                
                
    fh = mdptoolbox.mdp.FiniteHorizon(new_P, R, discount, stages)
    fh.run()
    value = fh.V[0][0]
    
    print(value)
    store_values[i][0] = np.linalg.norm(value-optimal_value)
                


# In[2]:


#Computing the best action (pivot) 
def get_pivot(state,states,actions):
    p1 = 0
    p2= 0
    for s2 in range(0,states):
        for a in range(0,actions):
#             print(p1,p2)
            count = 0
            prob = R[a,state,s2]/(R[a,state,s2] + R[p2,state,p1])
            for i in range(1000):
                count += np.random.choice(2,1,p = [prob,1-prob])
            if count < 490:
                p1 = s2
                p2 = a
    return [p1,p2]
                                


# In[3]:


#Computing the approximate rewards
def get_rewards(state,action,next_state,R):
    count = 1
    count_pivot = 1
    while count_pivot+count < 1000:
        p1 = R[action,state,next_state]/(1 + R[action,state,next_state])
        x = np.random.choice(2,1,p = [p1,1-p1])
        if x == 0:
            count +=1
        else:
            count_pivot +=1
    return count/count_pivot
    


# In[4]:


#Running UCFH algorithm 
epsilon_count = np.zeros((states,actions,states))
# constant = 3
# mod_R = R + 0.1/(constant*stages*(np.sqrt(states*states*actions)))
pivot1 = np.zeros((states,1))
pivot2 = np.zeros((states,1))

for s1 in range(states):
    x = get_pivot(s1,states,actions)
#     print(x)
    pivot1[s1,0] = x[0]
    pivot2[s1,0] = x[1]
    

mod_R = np.zeros((actions,states,states))
for s1 in range(states):
    for s2 in range(states):
        for a in range(actions):
            if a == pivot2[s1,0] and s2 == pivot1[s1,0]:
                mod_R[a,s1,s2] = 1
            else:
                mod_R[a,s1,s2] = get_rewards(s1,a,s2,R)

print('done')
for i in range(episodes):
    for j in range(sub_episodes):
        s = 0
        for k in range(stages):
            a = np.random.randint(actions)
            next_state = np.random.choice(states,1,p = P[a,s,:])[0]
            epsilon_count[s][a][next_state] += 1
            s = next_state
    
    new_P = np.zeros((actions,states,states))
    
    for a in range(actions):
        for s1 in range(states):
            if np.sum(epsilon_count[s1,a] >0):
                for s2 in range(states):
                    new_P[a,s1,s2] = epsilon_count[s1,a,s2]/(np.sum(epsilon_count[s1,a]))
            else:
                new_P[a,s1,s1] = 1
                
                
    fh = mdptoolbox.mdp.FiniteHorizon(new_P, mod_R, discount, stages)
    fh.run()
    value = fh.V[0][0]
    
    print(value)
    epsilon_store_values[i][0] = np.linalg.norm(value-optimal_value)
                


# In[12]:


print(store_values[499],epsilon_store_values[499])
# np.savetxt('actual_rewards',store_values)
# np.savetxt('modified_rewards',epsilon_store_values)


# In[9]:


import matplotlib.pyplot as plt
import pylab 
plt.plot(store_values,'b',linewidth = 2, label = 'Actual Rewards')
plt.plot(epsilon_store_values,'k',linewidth = 2, label = 'Modified Rewards')
pylab.axis([0, 500, 0, 1])
pylab.legend(loc = 'upper right',prop={'size': 14})
pylab.xlabel('Number of iterations',fontsize = 15)
pylab.ylabel('$|V^*(0) - V_{i}(0)|$',fontsize = 15)
pylab.show()


# In[7]:


np.linalg.norm(R-mod_R)


# In[8]:


for i in range(states):
    print(pivot1[i],pivot2[i],R[int(pivot2[i,0]),i,int(pivot1[i,0])])

