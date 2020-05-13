import numpy as np
import random
import uuid
import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding

#==================================== Document and Topic Model ====================================
sigma = 0.1
def lowQualityDoc():
	""" The remaining 14 topics are high quality,
	with their mean quality evenly distributed across the interval µt ∈ [−3, 0]  """
	mu_low=[random.uniform(-3,0) for i in range(14)]
	return [np.random.normal(mu_low[i], sigma, 1)[0] for i in range (len(mu_low))]

def hightQualityDoc():
	""" The remaining 6 topics are high quality,
	with their mean quality evenly distributed across the interval µt ∈ [0, 3]  """
	mu_hight=[random.uniform(0,3) for i in range(6)]
	return [np.random.normal(mu_hight[i], sigma, 1)[0] for i in range (len(mu_hight))]

def quality():
	a=lowQualityDoc()+hightQualityDoc()
	random.shuffle(a)
	return a

def choiceAleaList(tab,n):
  	return random.choice(tab)

topic=list(range(1,21)) #A list of topic we use 20 topics 

class Document:

	def __init__(self,ids,topic,length,inhQuality):
		self.id=ids
		self.topic=topic  #topic of document
		self.length=length #length of document
		self.inhQuality=inhQuality #inherted quality fo document
	
	'''def __str__(self):
    	return str(self.__class__) + ": " + str(self.__dict__)'''

def geNerNdocument(N):
	''' generate a set of N document  '''
	Doc=[0 for i in range(N)]
	for i in range(N):
		Doc[i]=Document(random.getrandbits(32),choiceAleaList(topic,20),4,choiceAleaList(quality(),20))
	return Doc

#==================================== User Interest and Satisfaction Models ====================================
interest=list(np.random.uniform(-1,1,20))  #Interest of user in different topic
	
class User :

	def __init__(self,ids,interests,age,sexe,lastRecom=0):
		self.id=ids
		self.lastRecom=lastRecom # user's last recommendation
		self.sexe=sexe
		self.age=age
		self.interests=interests #user'sinterest on topic of document

	'''def __str__(self):
    	return str(self.__class__) + ": " + str(self.__dict__)'''

def associateTopicInterest(user):
  ''' Topic =====> user's interest '''
  list1=topic
  list2=user.interests
  random.Random(4).shuffle(list1)
  random.Random(4).shuffle(list2)
  dico=dict(zip(list1,list2))
  dico.update({0:0})
  return dico
  
alpha=1 #We use an extreme value of α = 1.0 so that a user’s satisfaction 
#with a consumed document is fully dictated by documentquality. 

def userSatisfaction(user,document):
  ''' A user’s satisfaction S(u, d) with a consumed document d is a function f(I(u, d), Ld)
of user u’s interest and document d’s quality. While the form of f may be quite complex
in general, we assume a simple convex combination S(u, d) = (1 − α)I(u, d) + αLd.
 '''
  topicn=document.topic
  if topicn==len(topic):
    topicn=len(topic)-1
  return ((1-alpha)*associateTopicInterest(user)[topicn])+(alpha*document.inhQuality)

#==================================== User Choice Model ====================================
tau=0.01 #constante of the conditional logit

def generSlateofmDoc(m,user,alldoc):
	''' generate a slate of m best documents candidate from all Documents for a user in relation to the interest of the user  '''
	d=associateTopicInterest(user)
	A=sorted(d.items(), key=lambda x: x[1], reverse=True)
	kdoc=[]
	for j in range(m+1) :
		i=0
		while i < len(alldoc):
			if(alldoc[i].topic==A[j][0]):
				kdoc.append(alldoc[i])
				break
			else:
				i=i+1
	return kdoc

def generSlateofmDocAlea(m,alldoc):
	''' generate a slate of m  documents candidate randomly from all Documents for a user in relation to the interest of the user  '''
	return [random.choice(alldoc) for i in range (m)]

def generSlateofkDoc(k,alldoc):
	''' generate a slate of k documents candidate from slate of m documents generate for a user for recommendation '''
	return [random.choice(alldoc) for i in range (k)]
  

def featureVector(user,document):
	''' a set of user-item characteristics [userID,documentID,userInterst,documentTopic,documentLength,documentInhquality,usersatisfaction]'''
	topicn=document.topic
	if topicn==len(topic):
		topicn=len(topic)-1
	return [user.id,document.id,associateTopicInterest(user)[topicn],topicn,document.inhQuality,userSatisfaction(user,document)]

def unnormalizedprobability(user,document):
	''' unnormalized probability v(xij ) In the case of the conditional logit, v(xij ) = exp(τu(xij )), but any arbitrary v can be used '''
	if featureVector(user,document)[2]==0:
		return 0
	return math.exp(tau*userSatisfaction(user,document))

def somme(user,document):
	tab=[unnormalizedprobability(user,document[i]) for i in range(len(document)) ]
	return sum(tab)

#In our experiments, we use the general conditional choice
#model (Eq. (2)) as the main model for our RL methods

def conditionalLogitModel(user,document,slate):#(Eq. (2))
	''' The conditional logit model is an instance of a more general conditional choice format in
	which a user  selects document ∈ Documents with unnormalized probability  '''
	if featureVector(user,document)[2]==0:
		return 0
	return unnormalizedprobability(user,document)/somme(user,slate)

def addNullDoc(slate):
	slate.append(Document(000000,0,0,0))
	return slate

def secondChoicemodel():
	''' '''
	return 0

def combin(n, k):
    """Nombre de combinaisons de n objets pris k a k"""
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x

def allPossibleSlates(m,k,mdoc):
  """ generate all combinaisons of slate of k doc in m doc """
  total=combin(m,k)
  return [addNullDoc(random.choices(mdoc,k=k)) for i in range (total)]


		
#==================================== User Dynamics ====================================
y=0.3 #y ∈ [0, 1] denotes the fraction of the distance between the current interest level and the maximum level (1, −1) that the update
#move user u’s interest. In our experiemnt we set y=0.3

budget=200 #We assume each user u has an initial budget Bu of time to engage with content
#during an extended session
length=4 # document length
def bonus(user,document):
	""" We set bonus b =0.9/3.4 · l · S(u, d) after a user consume a document d"""
	return (0.9/3.4)*length*userSatisfaction(user,document)

def budgetAfterConsumDoc(user,document):
	''' The new budget of user after consuming an document '''
	global budget
	budget=budget-length+bonus(user,document)
	return budget

def changeInterestUser(user,document,slate):
	""" compute a new user's interest after consuming a document A positive change in interest, It ← It + ∆t(It),
	occurs with probability [I(u, d) + 1]/2, and a negative change, It ← It − ∆t(It), with
	probability [1 − I(u, d)]/2. """
	if conditionalLogitModel(user,document,slate)==0:
		return 0
	
	elif conditionalLogitModel(user,document,slate)>=((featureVector(user,document)[2]+1)/2):
		return featureVector(user,document)[2]+(-y*abs(featureVector(user,document)[2])+y)*-featureVector(user,document)[2]
		
	elif conditionalLogitModel(user,document,slate)<=((1-featureVector(user,document)[2])/2):
		return featureVector(user,document)[2]-(-y*abs(featureVector(user,document)[2])+y)*-featureVector(user,document)[2]

	else :
		return featureVector(user,document)[2]
	

	
class RecSys1(gym.Env):
	def __init__(self):
		self.user=User(random.getrandbits(16),list(np.random.uniform(-1,1,20)),random.choice([i for i in range(1,90)]),random.choice([i for i in range(1,3)]))
		self.alldocs=geNerNdocument(100)
		self.mdocs=generSlateofmDoc(10,self.user,self.alldocs)
		self.allslates=allPossibleSlates(10,3,self.mdocs)	
		self.budget=200
		self.historic=[]
		self.action_space=spaces.Discrete(3)
		self.observation_space=spaces.Discrete(combin(10,3))

	def next_Observation(self):
		self.slate=random.choice(self.allslates)
		self.choicedoc=random.choice(self.slate)
		self.historic.append(self.choicedoc.id)
		return self.choicedoc.id

	def _take_doc(self,action):
		self.slate=random.choice(self.allslates)
		self.choicedoc=random.choice(self.slate)
		self.historic.append(self.choicedoc.id)

	def step(self, action):
		self._take_doc(action)
		reward=bonus(self.user,self.choicedoc)
		self.budget=self.budget-self.choicedoc.length+reward
		done=self.budget<0
		obser=self.next_Observation()
		info={"Budgetafterconsumption":self.budget}
		return obser,reward,done,info

	def reset(self):
		self.budget=200
		self.historic=[]

	def render(self):
		topicn=self.choicedoc.topic
		if topicn==len(topic):
			topicn=len(topic)-1
		print("Document iD : ",self.choicedoc.id)
		print("Document's Topic : ",self.choicedoc.topic)
		print("Document's length : ",self.choicedoc.length)
		print("Document's Inherated Quality = user satisfaction beacause alpha=1 : ",self.choicedoc.inhQuality)
		print("user's interest before consuming document : ",self.user.interests[topicn])
		print("Logit Model : ",conditionalLogitModel(self.user,self.choicedoc,self.slate))
		print("user's interest after consuming document : ",changeInterestUser(self.user,self.choicedoc,self.slate))
		print("User's historic : ",self.historic)
