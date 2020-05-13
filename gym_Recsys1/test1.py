import gym
import random
import numpy as np
import gym_Recsys1
import time
env = gym.make('Recsys1-v0')
state_size = env.observation_space.n
action_size = env.action_space.n
Q_table = np.zeros((state_size, action_size))
rewards = []
MAX_EPISODES = 100
ALPHA = 0.8
GAMMA = 0.95
EPSILON = 1.0
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.005
for episode in range(MAX_EPISODES):
    S = env.reset()
    step = 0
    done = False
    total_rewards = 0

    while not done:
        # ETAPE 1
        if random.uniform(0, 1) < EPSILON:
            A = env.action_space.sample()
        else:
            A = np.argmax(Q_table[S, :])

        # ETAPE 2
        S_, R, done, info = env.step(A)

        # ETAPE 3
        q_predict = Q_table[S, A]
        if done:  # Si la partie est perdue ou gagnée, il n’y a pas d’état suivant
            q_target = R
        else:  # Sinon, on actualise et on continue
            q_target = R + GAMMA * np.max(Q_table[S_, :])
        Q_table[S, A] += ALPHA * (q_target - q_predict)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        total_rewards += reward

        env.render()
        

    EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
    rewards.append(total_rewards)
    print(total_rewards)