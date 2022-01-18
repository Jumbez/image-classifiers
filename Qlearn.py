# Jukka-Pekka Keinänen


# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time
from numpy.core.fromnumeric import argmax

# Environment
env = gym.make("Taxi-v3")

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Q table for rewards
Q_reward = numpy.random.random((500, 6)) # Random

# Training w/ random sampling of actions
def Q_learn(Q_table, a, gamma):
    for episode in range(num_of_episodes):
        state = env.reset()
        for step in range(num_of_steps):
            current_state = state
            action = numpy.argmax(Q_table[current_state,:])
            state, reward, done, info = env.step(action)
            Q_table[current_state, action] +=  a*(reward + gamma*numpy.amax(Q_table[state,:]) - Q_table[current_state, action])
            if done:
                break
            
Q_learn(Q_reward, alpha, gamma)

# Testing
all_tot_rewards = []
number_of_actions = []
for i in range(10):
    state = env.reset()
    tot_reward = 0
    numb_of_actions = 0
    for t in range(50):
        action = numpy.argmax(Q_reward[state,:])
        state, reward, done, info = env.step(action)
        tot_reward += reward
        numb_of_actions += 1
        env.render()
        time.sleep(1)
        if done:
            all_tot_rewards.append(tot_reward)
            number_of_actions.append(numb_of_actions)
            break

print(f"Average total reward {sum(all_tot_rewards)/len(all_tot_rewards)}.\nAverage number of steps {sum(number_of_actions)/len(number_of_actions)}")
