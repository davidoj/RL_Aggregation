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




class MountainCar():
    """
    The mountain car backend, with parameters as given by Sutton at
    http://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.lisp.
    """

        def __init__(self,randomStart=0,representation='raw',
                     aggregation=None,
                     numtilings=16,
                     divisions=100):

        assert representation in {'raw','disc','aggr','tile'}, "invalid representation requested"
        assert isinstance(randomStart,int), "invalid argument to randomStart"
        assert isinstance(aggregation,dict) or aggregation == None, "invalid aggregation"

        self.x = -0.5
        self.xMin = -1.2
        self.xGoal = 0.5

        self.xdot = 0
        self.xDotMax = 0.07
        
        self.dt = 1e-3
        self.grav = 2.5

        self.isEpisodic = 1
        self.gamma = 1
        self.randomStart = randomStart

        self.representation = representation
        self.aggregation = aggregation
        self.divisions = divisions

        self.actions = [-1,0,1]

        self.isEpisodic = True

        if representation == 'tile':
            self.numtilings = numtilings
            minMem = numtilings*self.divisions*self.divisions*10
            i = 1
            while 2**i < minMem:
                i += 1
            self.mem = 2**i

            self.ctable = tiles.CollisionTable(sizeval=self.mem)
        
    def reset(self):
        if self.randomStart:
            self.x = random.uniform(self.xMin,self.xGoal)
            self.xdot = random.uniform(-self.xDotMax, self.xDotMax)
        else:
            self.x = -0.5
            self.xdot = 0
    
    def getStateForAgent(self):
        if self.representation == 'raw':
            return self.x, self.xdot

        elif self.representation == 'disc':
            x_index = math.floor((self.x-self.xMin)*(self.divisions-1)/(self.xGoal-self.xMin))
            xdot_index = math.floor((self.xdot + self.xDotMax)*(self.divisions-1)/(2*self.xDotMax))
            return x_index + xdot_index

        elif self.representation == 'aggr':
            if self.x >= self.xGoal:
                return self.aggregation['term']
            x_index = math.floor((self.x-self.xMin)*(self.divisions-1)/(self.xGoal-self.xMin))
            xdot_index = math.floor((self.xdot + self.xDotMax)*(self.divisions-1)/(2*self.xDotMax))
            return self.aggregation[(xdot_index,x_index)]
 
        elif self.representation == 'tile':
            coords = (self.xDiv/(self.xGoal-self.xMin)*self.x, self.xDotDiv/(2*self.xDotMax)*self.xdot)
            return tiles.tiles(self.numtilings,self.ctable,coords)

    def bound_xdot(self):
        if abs(self.xdot)>self.xDotMax:
            self.xdot = np.copysign(self.xDotMax,self.xdot)
        
    def bound_x(self):
        if self.x<=self.xMin:
            self.xdot = 0
            self.x = self.xMin
        if self.x>=self.xGoal:
            self.xdot = 0
            self.x = self.xGoal


    def result(self,action):
        x = self.x
        xdot = self.xdot
        return self.getStateForAgent(), 

    def compute_result(self,action):
        action = self.actions[action]

        grav = - self.grav * np.cos(self.x*3)
        
        self.xdot += (action + grav)*self.dt
        self.bound_xdot()
        self.x += self.xdot
        self.bound_x()
