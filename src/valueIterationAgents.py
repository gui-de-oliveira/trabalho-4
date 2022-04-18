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

import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0

        for _ in range(iterations):
            self.update_state_values()

    def update_state_values(self):
        updated_values = self.values.copy()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            highest_action = {"action": None, "value": -999.0}

            for action in self.mdp.getPossibleActions(state):
                transitionsAndProbs = self.mdp.getTransitionStatesAndProbs(
                    state, action)

                value = 0.0
                for transition, prob in transitionsAndProbs:
                    value += prob * self.values[transition]

                if value > highest_action["value"]:
                    highest_action = {"action": action, "value": value}

            updated_values[state] = self.getReward(state) + \
                self.discount * highest_action["value"]

        self.values = updated_values

    # I dont know why action and nextState are needed to calculate the reward
    # So I made this function to simplify without breaking compatibility
    def getReward(self, state):
        return self.mdp.getReward(state, None, "")

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

        qvalue = 0.0

        for nextState, prob in self.mdp.getTransitionStatesAndProbs(
                state, action):
            next_state_value = self.values[nextState]
            qvalue += prob * next_state_value

        return self.getReward(state) + self.discount * qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        possibleActions = self.mdp.getPossibleActions(state)

        #  if there are no legal actions, which is the case at the terminal state, you should return None.
        if len(possibleActions) == 0:
            return None

        # The policy is the best action in the given state according to the values currently stored in self.values.
        highest_value_action = None
        highest_value = None

        for action in possibleActions:
            value = self.computeQValueFromValues(state, action)

            if highest_value_action == None or value > highest_value:
                highest_value_action = action
                highest_value = value
                continue

        return highest_value_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
