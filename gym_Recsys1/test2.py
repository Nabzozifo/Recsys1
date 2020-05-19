import gym
import random
import itertools 
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.style 
import numpy as np 
import pandas as pd 
import sys  
from collections import defaultdict 
import plottings  
matplotlib.style.use('ggplot') 
import gym_Recsys1
import time

env = gym.make('Recsys1-v0')
LEARNING_RATE= 0.1
DISCOUNT=0.95
EPISODES=500

Total_reward=[]
DISCRETRE_OS_SIZE=[20]*len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETRE_OS_SIZE
q_table=np.random.uniform(low=-10,high=10,size=env.action_space.n*4)

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def get_discrete_state(state):
	discrete_state=(state-env.observation_space.low)/discrete_os_win_size
	return  discrete_state.astype(np.int)  #tuple(tuple(a_m.tolist()) for a_m in discrete_state.astype(np.int) )


for episode in range(EPISODES):
	discrete_state = get_discrete_state(env.reset())
	done = False
	Totale_reward=0
	while not done:
		if np.random.random() > epsilon:
			# Get action from Q table
			action = np.argmax(q_table[discrete_state])
		else:
			# Get random action
			action = np.random.randint(0, env.action_space.n)


		new_state, reward, done, info = env.step(action)
		Totale_reward+=reward
		new_discrete_state = get_discrete_state(new_state)

		#print("New state : ",new_state,"\n")
		#env.render()
		#new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

		# If simulation did not end yet after last step - update Q table
		if not done:

			# Maximum possible Q value in next step (for new state)
			max_future_q = np.max(q_table[new_discrete_state])

			# Current Q value (for current state and performed action)
			current_q = q_table[discrete_state + (action,)]

			# And here's our equation for a new Q value for current state and action
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

			# Update Q table with new Q value
			q_table[discrete_state + (action,)] = new_q

		discrete_state = new_discrete_state
	
	Total_reward.append(Totale_reward)
	# Decaying is being done every episode if episode number is within decaying range
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value


env.close()

print("Total reward : ",sum(Total_reward))
print("Average reward : ",sum(Total_reward)/len(Total_reward))
plt.plot(Total_reward)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

import csv

with open('q_value.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile)
     wr.writerow(q_table)


'''print(q_table)
print(len(q_table))'''
"""Evaluate agent's performance after Q-learning"""

total_epochs=0
episodes = 50
Reward=[]
total_nb_nul_doc=0
total_nb_pass_doc=0
total_nb_consomdoc=0

for _ in range(episodes):
   
	discrete_state = get_discrete_state(env.reset())
	
	nb_consomdoc=0
	nb_nul_doc=0
	nb_pass_doc=0
	epochs= 0
	total_reward = 0
	
	
	done = False
	
	while not done:
		action = np.argmax(q_table[discrete_state])
		state, reward, done, info = env.step(action)
		total_reward+=reward
		'''if reward==0 :
			nb_pass_doc+=1
		elif reward==1:
			nb_nul_doc+=1
		else :
			nb_consomdoc+=1'''

		epochs += 1
	Reward.append(total_reward)
	total_nb_nul_doc+=nb_nul_doc
	total_nb_pass_doc+=nb_pass_doc
	total_nb_consomdoc+=nb_consomdoc
	total_epochs += epochs
'''print("Total Documents consomés : ",total_nb_consomdoc)
print("Total Documents que utilisateur les a ignoré : ",total_nb_pass_doc)
print("Total Documents non consommés (Document null) : ",total_nb_nul_doc)'''
print("\nTESSSSSTTTTIIIIINNNNNNNNNNNG")
print("Average Reward : ",sum(Reward)/len(Reward))
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")

'''for _ in range(10):
	action = env.action_space.sample()
	print(env.step(action))'''
'''state=env.reset()
print(env.action_space.n)
print(get_discrete_state(state))

print(get_discrete_state(state)+(action,))
print(q_table[get_discrete_state(state)+(action,)])
print(np.argmax(q_table[get_discrete_state(state)]))'''