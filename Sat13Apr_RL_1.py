'''
Thanks to morvanZhou
An agent is on the left, and the treasure is in the very right.
The environment is the 1d
'''

import numpy as np
import pandas as pd
import time

np.random.seed(123) #reproducible

N_STATES = 6 #the length of the 1d world
ACTIONS = ['left','right']
EPSILON = 0.9 # epsilon greedy
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor
MAX_EPISODES = 13 # maximum episodes
FRESH_TIME = 0.3 # fresh time for one move(dont know what this is 'for now')

def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states,len(actions))), columns = actions)
    print(table)
    return table

def choose_actions(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform()<EPSILON) or (state_actions == 0).all():
        action_name = np.random.choice(ACTIONS)
    else: #act greedy
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S,A): #retuns state and reward
    if A == 'right':
        if S == N_STATES -2:
            print('Goal In!') # from 0 to 5
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R = 0
    else: # a is left
        R = 0
        if S == 0:
            S_ = S
            print('No where to go left, only wall')
        else:
            S_ = S -1
    return S_,R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T'] # T: terminal
    if S == 'terminal':
        interaction = 'Episode %s: total_steps %s' %(episode+1, step_counter)
        print('\r{}'.format(interaction), end = '')
        time.sleep(2)
        print('\r                                    ', end = '')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end = '')
        time.sleep(FRESH_TIME) #0.3

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_actions(S, q_table)
            S_,R = get_env_feedback(S,A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R+GAMMA*q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S,A] += ALPHA * (q_target-q_predict) # update
            S = S_

            update_env(S, episode, step_counter+1)
            step_counter += 1
    print(q_table)
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
