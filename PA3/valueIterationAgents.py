# valueIterationAgents.py
# -----------------------
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

import random
import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.value_itration()

    # Write value iteration code here
    "*** YOUR CODE HERE ***"
    def value_itration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            update_values = self.values.copy()
            for state in self.mdp.getStates():
                q_values = []
                isTerminal = self.mdp.isTerminal(state)  

                if isTerminal:
                    update_values[state] = 0

                else:
                    legal_actions = self.mdp.getPossibleActions(state)

                    for action in legal_actions:
                        q_values.append(self.getQValue(state, action))
                    update_values[state] = max(q_values)

            self.values = update_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """

        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transition_states = self.mdp.getTransitionStatesAndProbs(state, action) 
        q_values = []
        for next_state, prob in transition_states:
            r = self.mdp.getReward(state, action, next_state)
            gamma = self.discount * self.values[next_state]
            q = prob * (r + gamma) 
            q_values.append(q)
        s_q =  sum(q_values)
        return s_q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state) # Retrieving Possible Actions

        if len(actions)==0:
            return None #Returning None if there are no possible actions
        action_QValue_pairs = {} 

        #Creating a dictionary action value pairs 
        for a in actions:
          action_QValue_pairs[a] = self.getQValue(state,a)
        max_value = max(action_QValue_pairs.values())
        

        actions = [k for k,v in action_QValue_pairs.items() if v == max_value]
        
        #Randomly choosing a max value
        max_key = random.choice(actions)
        return max_key


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
