'''
Reinforcement Learning Problems

David Johnston 2015
'''


import math
import numpy as np
import random
from datetime import datetime
#from Tiles import tiles
import Random_MDP
#import Aggregation

def genRandomProblems(n,n_agg,actions,neighbours,gamma,aggType='q'):
    '''
    Returns an aggregated and raw problem instance with 
    same underlying MDP and n, n_agg states respectively
    '''
    transitions = np.zeros((actions,n,n))
    
    values = Random_MDP.getValues(n,n_agg,actions,aggType=aggType)
    for i in range(len(transitions)):
        transitions[i] =  Random_MDP.getTrns(n,neighbours)
    rewards = Random_MDP.getRewards(transitions,values,gamma)
    aggregation = Random_MDP.getAggregation(values[0])

    p_raw = MDP(0,transitions,rewards,gamma,qValues=values)
    p_agg = MDP(0,transitions,rewards,gamma,aggregation=aggregation,qValues=values)

    return p_raw, p_agg


    
class MDP:
    """
    An MDP. Contains methods for initialisation, state transition. 
    Can be aggregated or unaggregated.
    """

    def __init__(self, initial, transitions, rewards, gamma, aggregation = None,
                 qValues = None):       
        self.actions = range(len(transitions))
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma

        self.qValues = qValues

        self.reset = None
        self.isEpisodic = 0
        self.aggregation = aggregation

        self.rawStates = len(transitions[0])
        if isinstance(aggregation,dict):
            self.nStates = aggregation['n']
        else:
            self.nStates = self.rawStates
        self.problemState = initial

        d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
        b = sum([1 for x in transitions[0][0] if x>0])
        self.probName = "{}_n={}_b={}_gamma={}_agg={}".format(d,len(transitions[0]),
                                                              b,gamma,self.nStates)

        
    def getZeroQTable(self):
        return np.zeros((self.nStates,len(self.actions)))
        
    def setAlpha(self, alpha):
        return alpha
    
    def getActions(self):
        return self.actions
        
    def getAgentState(self):
        if self.aggregation:
            return self.aggregation[self.problemState]
        else:
            return self.problemState
 
            
    def result(self, action):

        stateVec = np.zeros(self.rawStates)
        stateVec[self.problemState] = 1
        
        successorVec = stateVec.dot(self.transitions[action])

        successor = np.random.choice(self.rawStates,1,p=successorVec)[0]
        
        reward = self.rewards[action][self.problemState][successor]

        self.problemState = successor           

        if self.aggregation:
            agg_successor = self.aggregation[successor]
            return agg_successor, reward
            
        else:
            return successor, reward         
