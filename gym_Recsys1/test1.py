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
#print(rcs.geNerNuser(10))
random.seed(30)
users=rcs.geNerNuser(10)
docs=rcs.geNerNdocument(100)
docu=rcs.Document(1111,1,4,5.22589655899)
docs.append(docu)
#env = gym.make('Recsys1-v0',user=users[1],alldocs=docs)
rs=open("result.txt",'w')
plt.figure(figsize=(20,10))
import seaborn as sns
for i in range(1,4):
	env = gym.make('Recsys1-v0',user=users[i],alldocs=docs)
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
	
	rs.write("Average Reward : "+ str(sum(Reward)/len(Reward))+"\n")
	from collections import Counter
	rs.write("user : "+ str(users[i])+"\n")
	rs.write("total document consomme : "+str(len(doc_consume))+"\n")
	z=Counter(doc_consume)
	y=Counter([doc_consume[k].id for k in range(len(doc_consume)) ])
	rs.write("les 20 Documents les plus consomemer : "+"\n")
	rest=sorted(z.items(), key=lambda x: x[1],reverse=True)
	reste=sorted(y.items(), key=lambda x: x[1],reverse=True)
	# save the names and their respective scores separately
	# reverse the tuples to go from most frequent to least frequent 
	doc = list(zip(*reste))[0]
	#print(doc[1].id)
	#docu=[doc[k].id for k in range(len(doc))]
	consom = list(zip(*reste))[1]
	x_pos = np.arange(len(doc)) 

	# calculate slope and intercept for the linear trend line
	slope, intercept = np.polyfit(x_pos, consom, 1)
	trendline = intercept + (slope * x_pos)

	#plt.plot(x_pos, trendline, color='red', linestyle='--')    
	plt.bar(x_pos, consom,align='center')
	plt.xticks(x_pos, doc) 
	plt.ylabel('Nb Consum Doc')
	plt.savefig("ConsumDoc_"+"user_"+str(i))
	plt.clf()
	#rest2=sorted(z.keys(), key=lambda x: x[1],reverse=True)
	for j in range(20):
		rs.write("Documents : "+ rest[j][0].__str__()+"Nombre de fois consommes : "+str(rest[j][1])+"\n")
	plt.plot(LTV)
	plt.savefig("ltv_"+"user_"+str(i))
	plt.clf()
	#plottings.plot_episode_stats(stats) 