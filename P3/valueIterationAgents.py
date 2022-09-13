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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            tmp = util.Counter()
            for state in self.mdp.getStates():
                tmp[state] = self.computeActionFromValues(state)
            for state in self.mdp.getStates():
                if tmp[state] is None:
                    tmp[state] = 0
                    continue
                tmp[state] = self.computeQValueFromValues(state, tmp[state])
            self.values = tmp

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
        list_of_state_ans_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        val = 0
        for nextState, p in list_of_state_ans_prob:
            val += p * (self.mdp.getReward(state, action, nextState) +
                            self.discount * self.getValue(nextState))
        return val


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state): return None
        actions = self.mdp.getPossibleActions(state)
        val = -99999
        ans = None
        for action in actions:
            tmp_val = self.computeQValueFromValues(state, action)
            if tmp_val > val:
                val = tmp_val
                ans = action
        return ans


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        i = 0
        while i != self.iterations:
            for state in self.mdp.getStates():
                action = self.computeActionFromValues(state)
                if action is not None:
                    self.values[state] = self.computeQValueFromValues(state, action)
                i += 1
                if i == self.iterations : return


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}
        pq = util.PriorityQueue()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            maxQ = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            diff = abs(maxQ - self.values[s])
            pq.update(s, -diff)
        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            for p in predecessors[s]:
                maxQ = max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)])
                diff = abs(maxQ - self.values[p])
                if diff > self.theta:
                    pq.update(p, -diff)

        """
        # Compute predecessors of all states
        predecessors = dict()
        for state in self.mdp.getStates():
            action = self.computeActionFromValues(state)
            if action is not None:
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob == 0 or self.mdp.isTerminal(nextState): continue
                    if nextState not in predecessors.keys():
                        tmp = set()
                        tmp.add((state[0], state[1]))
                        predecessors[nextState] = tmp
                    else:
                        predecessors[nextState].add((state[0], state[1]))
        # Initialize an empty priority queue
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state): continue
            action = self.computeActionFromValues(state)
            if action is None: continue
            diff = abs(self.getValue(state) - (self.computeQValueFromValues(state, action)))
            pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty(): break
            state = pq.pop()
            # if not self.mdp.isTerminal(state):
            act = self.computeActionFromValues(state)
            self.values[state] = self.computeQValueFromValues(state, act)
            if state not in predecessors.keys(): continue
            for predecessor in predecessors[state]:
                action = self.computeActionFromValues(predecessor)
                diff = abs(self.getValue(predecessor) - (self.computeQValueFromValues(predecessor, action)))
                if diff > self.theta:
                    pq.update(predecessor, -diff)

        """