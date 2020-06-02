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
#env.env.user(users[1])
LEARNING_RATE= 0.1
DISCOUNT=0.95
EPISODES=10
Total_reward=[]
DISCRETRE_OS_SIZE=[20]*len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETRE_OS_SIZE
q_table=np.zeros(env.action_space.n*4)
ltv_table=np.zeros(env.action_space.n*4)
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
		if random.random() > epsilon:
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
			max_future_q = np.argmax(q_table[new_discrete_state])
			future_ltv = ltv_table[new_discrete_state]
			

			# Current Q value (for current state and performed action)
			current_q = q_table[discrete_state + (action,)]
			current_ltv=ltv_table[discrete_state + (action,)]

			# And here's our equation for a new Q value for current state and action
			new_ltv = (1 - LEARNING_RATE) * current_ltv + LEARNING_RATE * (reward + DISCOUNT * future_ltv)
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

			# Update Q table with new Q value
			ltv_table[discrete_state + (action,)] = new_ltv
			q_table[discrete_state + (action,)] = new_q

		discrete_state = new_discrete_state
	
	Total_reward.append(Totale_reward)
	# Decaying is being done every episode if episode number is within decaying range
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value




#print("LTV TABLE : ",ltv_table)
#print("Q TABLE : ",q_table)
#print("Total reward : ",sum(Total_reward))
#print("Average reward : ",sum(Total_reward)/len(Total_reward))
'''import seaborn as sns
plt.plot([i for i in range(150)],[ltv_table[i] for i in range(150)])
plt.xlabel("Episode")
plt.ylabel("LTV")
plt.show()'''
#print(Total_reward)

total_epochs=0
episodes = 10
Reward=[]
doc_consume=[]

LTV=[]

for _ in range(episodes):
	discrete_state = get_discrete_state(env.reset())
	epochs= 0
	total_ltv=0
	total_reward = 0
	done = False
	while not done:
		action = np.argmax(q_table[discrete_state])
		ltv=np.max(q_table[discrete_state])
		state, reward, done, info = env.step(action)
		total_reward+=reward
		total_ltv+=ltv
		epochs += 1
	Reward.append(total_reward)
	LTV.append(total_ltv)
	total_epochs += epochs
	doc_consume+=env.historic
from collections import Counter
print("user : ",users[1])
print("total document consomme : ",len(doc_consume))
z=Counter(doc_consume)
print("les 10 Documents les plus consomemer : ")
rest=sorted(z.items(), key=lambda x: x[1],reverse=True)
#rest2=sorted(z.keys(), key=lambda x: x[1],reverse=True)
for i in range(10):
	print("Documents : ",rest[i][0].__str__(),"Nombre de fois consommes : ",rest[i][1])
print("Average Reward : ",sum(Reward)/len(Reward))
print(LTV)
plt.plot(LTV)
plt.legend()
plt.show()
plt.plot(Reward)
plt.legend()
plt.show()

env.close()