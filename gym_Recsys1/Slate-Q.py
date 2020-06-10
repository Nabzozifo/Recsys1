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
u1=rcs.User(1000,20,rcs.associateTopicInterest(),1,10000)
u2=rcs.User(1001,21,rcs.associateTopicInterest(),2,10000)
u3=rcs.User(1002,22,rcs.associateTopicInterest(),1,10000)
u4=rcs.User(1003,30,rcs.associateTopicInterest(),2,20000)
u5=rcs.User(1004,31,rcs.associateTopicInterest(),1,20000)
u6=rcs.User(1005,32,rcs.associateTopicInterest(),2,20000)
u7=rcs.User(1006,50,rcs.associateTopicInterest(),1,30000)
u8=rcs.User(1007,51,rcs.associateTopicInterest(),2,30000)
u9=rcs.User(1008,52,rcs.associateTopicInterest(),1,30000)
users=[u1,u2,u3,u4,u5,u6,u7,u8,u9]
#users=rcs.geNerNuser(10)
#random.seed(30)
docs=rcs.geNerNdocument(50)
docu=rcs.Document(888,1,4,10.22589655899)
docs.append(docu)
plt.figure(figsize=(20,10))
import seaborn as sns
def get_discrete_state(state):
	discrete_state=(state-env.observation_space.low)/discrete_os_win_size
	return  discrete_state.astype(np.int)

def ltver(usere,slate,q):
	a=[]
	for doci in slate:
		somme=0
		for doce in slate:
			b=rcs.conditionalLogitModel(usere,doce,slate)
			somme=somme+b*q
		a.append(somme)
	return a

for i in range(len(users)):
	rs=open("result/result_"+"user_"+str(i)+".txt",'w')
	rs.write("Interest user before consume docs : "+ str(sorted(users[i].associate_topic_interet.items(), key=lambda x: x[1], reverse=True))+"\n")
	env = gym.make('Recsys1-v0',user=users[i],alldocs=docs)
	#env.env.user(users[1])
	LEARNING_RATE= 0.1
	DISCOUNT=1
	EPISODES=100
	Total_reward=[]
	DISCRETRE_OS_SIZE=[20]*len(env.observation_space.high)
	discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETRE_OS_SIZE
	q_table=np.zeros(env.action_space.n*5)
	ltv_table=np.zeros(env.action_space.n*5)
	# Exploration settings
	epsilon = 1  # not a constant, qoing to be decayed
	START_EPSILON_DECAYING = 1
	END_EPSILON_DECAYING = EPISODES//2
	epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
  #tuple(tuple(a_m.tolist()) for a_m in discrete_state.astype(np.int) )


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
	LTV2=[]

	for _ in range(episodes):
		discrete_state = get_discrete_state(env.reset())
		epochs= 0
		total_ltv=0
		total_ltv_2=0
		total_reward = 0
		done = False
		while not done:
			action = np.argmax(q_table[discrete_state])
			ltv=np.max(q_table[discrete_state])
			state, reward, done, info = env.step(action)
			total_reward+=reward
			total_ltv_2+=np.max(ltver(users[i],info["Slate"],np.max(q_table[discrete_state])))
			total_ltv+=ltv
			epochs += 1
		Reward.append(total_reward)
		LTV.append(total_ltv)
		LTV2.append(total_ltv_2)
		#print(info["Slate"])
		#print("logitmodel",rcs.conditionalLogitModel(users[i],info["doc"],info["Slate"]))
		#print(ltver(users[i],info["Slate"],np.max(q_table[discrete_state])))
		#print(np.max(q_table[discrete_state]))
		total_epochs += epochs
		doc_consume+=env.historic
	rs.write("Average Reward : "+ str(sum(Reward)/len(Reward))+"\n")
	from collections import Counter
	rs.write("user after consume docs : "+ str(sorted(users[i].associate_topic_interet.items(), key=lambda x: x[1], reverse=True))+"\n")
	rs.write("total document consomme : "+str(len(doc_consume))+"\n")
	z=Counter(doc_consume)
	y=Counter([doc_consume[k].id for k in range(len(doc_consume)) ])
	rs.write("Les Documents consommés : "+"\n")
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
	plt.xlabel('Doc ID')
	plt.ylabel('Nb Consum Doc')
	plt.title("ConsumDoc_"+"user_"+str(i))
	plt.savefig("result/ConsumDoc_"+"user_"+str(i))
	plt.clf()
	#rest2=sorted(z.keys(), key=lambda x: x[1],reverse=True)
	for j in range(len(rest)):
		rs.write("Documents : "+ rest[j][0].__str__()+"Nombre de fois consommes : "+str(rest[j][1])+"\n")
	total_quality=sum([doc_.inhQuality for doc_ in doc_consume])
	rs.write("Average qality of documents consume by user : "+str(users[i].id)+"is : "+str(total_quality/len(doc_consume)) +"\n")
	rs.close
	plt.plot(LTV)
	plt.title("user_"+str(i)+" engagement")
	plt.savefig("result/ltv1_"+"user_"+str(i))
	plt.clf()
	"""plt.plot(LTV2)
	plt.savefig("result/ltv2_"+"user_"+str(i))
	plt.clf()"""
	
	env.close()
	