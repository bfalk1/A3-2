# solutions.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

'''Implement the methods from the classes in inference.py here'''


import util
from util import raiseNotDefined
import random
import busters

def normalize(self):
    """
    Normalize the distribution such that the total value of all keys sums
    to 1. The ratio of values for all keys will remain the same. In the case
    where the total value of the distribution is 0, do nothing.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> dist.normalize()
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
    >>> dist['e'] = 4
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
    >>> empty = DiscreteDistribution()
    >>> empty.normalize()
    >>> empty
    {}
    """
    totalKeyValue = sum(self.values()) #Create sum
    if totalKeyValue != 0: #Check if normalization is needed
        for key in self:
            self[key] /= totalKeyValue #normalize

def sample(self):
    """
    Draw a random sample from the distribution and return the key, weighted
    by the values associated with each key.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> N = 100000.0
    >>> samples = [dist.sample() for _ in range(int(N))]
    >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
    0.2
    >>> round(samples.count('b') * 1.0/N, 1)
    0.4
    >>> round(samples.count('c') * 1.0/N, 1)
    0.4
    >>> round(samples.count('d') * 1.0/N, 1)
    0.0
    """
    total_sum = sum(self.values()) #Get sum
    randomVal = random.random() * total_sum #Create random value
    cumulativeSum = 0
    for key, value in self.items():
        cumulativeSum += value #Increment by random value to 
        if cumulativeSum > randomVal: #Sum is greater than random val => return corresponding key 
            return key

from util import manhattanDistance

def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
    """
    Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
    """
    if ghostPosition == jailPosition:
       #If the ghost pos is the same as the jail pos it is either one or the ghost is 
        #not observed 
        return 1 if noisyDistance is None else 0

   
    if noisyDistance is None:
        #ghost is not observed so return zero
        return 0

   
    trueDistance = manhattanDistance(pacmanPosition, ghostPosition) #Calculate the mahanttan distance

    
    return busters.getObservationProbability(noisyDistance, trueDistance)



def observeUpdate(self, observation, gameState):
    """
    Update beliefs based on the distance observation and Pacman's position.

    The observation is the noisy Manhattan distance to the ghost you are
    tracking.

    self.allPositions is a list of the possible ghost positions, including
    the jail position. You should only consider positions that are in
    self.allPositions.

    The update model is not entirely stationary: it may depend on Pacman's
    current position. However, this is not a problem, as Pacman's current
    position is known.
    """
    pac_pos = gameState.getPacmanPosition() #Get pac position
    jail_pos = self.getJailPosition() #get jail position

   
    if observation is None:
        
        self.beliefs = util.Counter({jail_pos: 1.0}) #If the ghost is not observed init beliefs
    else: #Else iterate over all positible ghost positions and calculate the probability
        for ghost_pos in self.allPositions:
            observation_prob = self.getObservationProb(observation, pac_pos,
                                                      ghost_pos, jail_pos)
            self.beliefs[ghost_pos] *= observation_prob

    self.beliefs.normalize()


def elapseTime(self, gameState):
    """
    Predict beliefs in response to a time step passing from the current
    state.

    The transition model is not entirely stationary: it may depend on
    Pacman's current position. However, this is not a problem, as Pacman's
    current position is known.
    """
  
    updated_beliefs = util.Counter() #Init beliefs array

   
    for old_pos in self.allPositions: #Iterate over all positions

       
        new_pos_distribution = self.getPositionDistribution(gameState, old_pos) #Get distribution of each position

       
        for new_pos, prob in new_pos_distribution.items():
            updated_beliefs[new_pos] += self.beliefs[old_pos] * prob #Update beliefs array

   
    updated_beliefs.normalize() #Normalize if needed

   
    self.beliefs = updated_beliefs #Update class property

