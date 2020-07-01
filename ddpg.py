import gym
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network_bn import CriticNetwork
from actor_network_bn import ActorNetwork
from replay_buffer import ReplayBuffer
from dm_control import suite
from numpy import array, exp

REPLAY_BUFFER_SIZE = 10000000
REPLAY_START_SIZE = 128
BATCH_SIZE = 128
GAMMA = 0.81

def obs2state(observation):
    l1 = [val.tolist() for val in list(observation.values())]
    l2 = []
    for sublist in l1:
        try:
            l2.extend(sublist)
        except:
            l2.append(sublist)
    return l2

class DDPG:
    def __init__(self, env):
        self.name = 'DDPG'
        self.environment = env
        self.episode = 0
        self.epsilon = 0.98
        self.one_number = 1
        self.mean = []
        self.state_dim = len(obs2state(env.reset().observation))
        self.action_dim = env.action_spec().shape[0]

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim)
        
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
         
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []  
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        self.critic_network.train(y_batch,state_batch,action_batch)
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self,state):
        action = self.actor_network.action(state)
        exp = self.exploration_noise.noise()
        t = action*exp
        return exp

    def action(self,state):
        if np.random.rand() <= self.epsilon:
            act = self.noise_action(state)
            z = array(act)
        else:
            action = self.actor_network.action(state)
            z = array(action)
        self.mean.append(z[0])
        g = np.tanh(z)
        return g

    def perceive(self,state,action,reward,next_state,done):
        self.replay_buffer.add(state,action,reward,next_state,done)
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.train()
        if self.epsilon > 0.1:
            self.epsilon *= 0.99999
            
        if done:
            self.exploration_noise.reset()










