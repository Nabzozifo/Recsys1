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
import sys
sys.path.append('gym_Recsys1/gym_Recsys1/envs/')
import Recsys1_env as rcs

#print(rcs.geNerNuser(10))
#random.seed(30)
users=rcs.geNerNuser(10)
docs=rcs.geNerNdocument(100)
env = gym.make('Recsys1-v0',user=users[1],alldocs=docs)

def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
	""" 
	Creates an epsilon-greedy policy based 
	on a given Q-function and epsilon. 
	   
	Returns a function that takes the state 
	as an input and returns the probabilities 
	for each action in the form of a numpy array  
	of length of the action space(set of possible actions). 
	"""
	def policyFunction(state): 
   
		Action_probabilities = np.ones(num_actions, 
				dtype = float) * epsilon / num_actions 
				  
		best_action = np.argmax(Q[state]) 
		Action_probabilities[best_action] += (1.0 - epsilon) 
		return Action_probabilities 
   
	return policyFunction

def qLearning(env, num_episodes, discount_factor = 1.0, 
							alpha = 0.6, epsilon = 0.1): 
	""" 
	Q-Learning algorithm: Off-policy TD control. 
	Finds the optimal greedy policy while improving 
	following an epsilon-greedy policy"""
	
	# Action value function 
	# A nested dictionary that maps 
	# state -> (action -> action-value). 
	Q = defaultdict(lambda: np.zeros(env.action_space.n*4)) 

	# Keeps track of useful statistics 
	stats = plottings.EpisodeStats( 
		episode_lengths = np.zeros(num_episodes), 
		episode_rewards = np.zeros(num_episodes))	 
	
	# Create an epsilon greedy policy function 
	# appropriately for environment action space 
	policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n*4) 
	
	# For every episode 
	for ith_episode in range(num_episodes): 
		state = env.reset()
		# Reset the environment and pick the first action 
		for t in itertools.count(): 
			
			# get probabilities of all actions from current state 
			action_probabilities = policy(state) 

			# choose action according to 
			# the probability distribution 
			action = np.random.choice(np.arange( 
					len(action_probabilities)), 
					p = action_probabilities) 

			# take action and get reward, transit to next state 
			next_state, reward, done, _ = env.step(action) 

			# Update statistics 
			stats.episode_rewards[ith_episode] += reward 
			stats.episode_lengths[ith_episode] = t 
			
			# TD Update 
			
			best_next_action = np.argmax(Q[next_state])	 
			td_target = reward + discount_factor * Q[next_state][best_next_action] 
			td_delta = td_target - Q[state][action] 
			Q[state][action] += alpha * td_delta 

			# done is True if episode terminated 
			if done: 
				break
				
			state = next_state 
	
	return Q,stats

q_table,stats=qLearning(env, 100) 
#print(q_table)
#plottings.plot_episode_stats(stats)

"""Evaluate agent's performance after Q-learning"""

total_epochs=0
episodes = 10
Reward=[]
total_nb_nul_doc=0
total_nb_pass_doc=0
total_nb_consomdoc=0
LTV=[]
doc_consume=[]
for _ in range(episodes):
   
	state = env.reset()
	nb_consomdoc=0
	nb_nul_doc=0
	nb_pass_doc=0
	epochs= 0
	total_reward = 0
	total_ltv=0
	
	done = False
	
	while not done:
		action = np.argmax(q_table[state])
		ltv=np.max(q_table[state])
		state, reward, done, info = env.step(action)
		total_reward+=reward
		total_ltv+=ltv
		epochs += 1
	Reward.append(total_reward)
	LTV.append(total_ltv)
	total_nb_nul_doc+=nb_nul_doc
	total_nb_pass_doc+=nb_pass_doc
	total_nb_consomdoc+=nb_consomdoc
	total_epochs +=epochs 
	doc_consume+=env.historic
print("Average Reward : ",sum(Reward)/len(Reward))
from collections import Counter
print("user : ",users[1])
print("total document consomme : ",len(doc_consume))
z=Counter(doc_consume)
print("les 10 Documents les plus consomemer : ")
rest=sorted(z.items(), key=lambda x: x[1],reverse=True)
#rest2=sorted(z.keys(), key=lambda x: x[1],reverse=True)
for i in range(10):
	print("Documents : ",rest[i][0].__str__(),"Nombre de fois consommes : ",rest[i][1])
plt.plot(LTV)
plt.show()
#plottings.plot_episode_stats(stats) 