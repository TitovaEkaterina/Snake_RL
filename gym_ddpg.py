import filter_env
from ddpg import *
import gc
from collections import deque
from dm_control import suite
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from IPython import display
from dm_control_to_gym import Env_DM_Control
from dm_control import viewer
gc.enable()
env = suite.load(domain_name="swimmer", task_name="swimmer6")

ENV_NAME = 'Swimmer-v3'
EPISODES = 1400

def obs2state(observation):
    l1 = [val.tolist() for val in list(observation.values())]
    l2 = []
    for sublist in l1:
        try:
            l2.extend(sublist)
        except:
            l2.append(sublist)
    return l2
    
if __name__ == '__main__':
    
    agent = DDPG(env)
    
    loss = []
    Y = []
    resultLast100Episodes = []
    X = []
    
    for episode in range(EPISODES):
        if episode > 1350:
            def policy(time_step):
                state = obs2state(time_step.observation)
                rewardd = time_step.reward
                action = agent.action(state)
                return action
#
            viewer.launch(env, policy=policy)
        currentReward = 0
        currentReward2 = 0
        time_step = env.reset()
        state = obs2state(time_step.observation)
        # Train
        vid = 0
        lastRevard = 0
        vid = 0
        ttt = 0
        ss = 0
        for step in range(1000):
            total_reward = 0
            action = agent.action(state)
            time_step = env.step(action)
            
            reward = time_step.reward
            nextState = obs2state(time_step.observation)
            state = nextState
            terminal = time_step.last()

            currentReward += reward
            
            tempRevard = (reward - lastRevard)*100

            ttt = (currentReward/(step+1) - lastRevard)
            ss = step+1
            
            if reward == 1:
                ttt = ttt + 1
                
            agent.perceive(state,action,ttt,nextState,terminal)
            # lastRevard = reward
            lastRevard = currentReward/(step+1)
            if terminal or reward == 1:
                if reward == 1:
                    currentReward = 1000
                break
        loss.append(currentReward)
        is_solved = np.mean(loss[-10:])
        Y.append(is_solved)
        resultLast100Episodes.append(is_solved)
        X.append(episode)
        
        num_bins = 25
        fig, axs = plt.subplots(2)
         
        n, bins, patches = axs[0].hist(resultLast100Episodes, num_bins, facecolor='blue', alpha=0.5)
         
        axs[1].plot(X, Y)
         
        plt.savefig("resultHist3.png")
        
        print('episode: ', episode, 'SCORE: ',currentReward,'MEAN:', is_solved, 'epsilon: ', agent.epsilon, 'step: ', ss)
        
