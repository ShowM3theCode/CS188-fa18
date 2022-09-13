# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if newFood.count() == 0: return 99999
        minFoodDistance = 99999
        newFood = newFood.asList()
        for food in newFood:
            minFoodDistance = min(minFoodDistance, util.manhattanDistance(newPos, food))
        minGhostDistance = 99999
        GhostPos = successorGameState.getGhostPositions()
        for pos in GhostPos:
            minGhostDistance = min(minGhostDistance, util.manhattanDistance(pos, newPos))
        if minGhostDistance <= 1: return -99999
        return successorGameState.getScore() + 1 / minFoodDistance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """


    def value(self, state, depth, agentNum):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if depth == self.depth + 1:
            return self.evaluationFunction(state)
        if agentNum == 0:
            return self.max_value(state, depth, agentNum)
        else:
            return self.min_value(state, depth, agentNum)

    def max_value(self, state, depth, agentNum):
        v = -99999
        for action in state.getLegalActions(agentNum):
            successor = state.generateSuccessor(agentNum, action)
            v = max(v, self.value(successor, depth, agentNum + 1))
        return v

    def min_value(self, state, depth, agentNum):
        v = 99999
        for action in state.getLegalActions(agentNum):
            successor = state.generateSuccessor(agentNum, action)
            if agentNum == state.getNumAgents() - 1:
                v = min(v, self.value(successor, depth + 1, 0))
            else:
                v = min(v, self.value(successor, depth, agentNum + 1))
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        v = -99999
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            val = self.value(successor, 1, 1)
            if val > v:
                v = val
                move = action
        return move

'''
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, state, depth, agentNum, alpha, beta):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        if depth == self.depth + 1:
            return self.evaluationFunction(state), None
        if agentNum == 0:
            return self.max_value(state, depth, agentNum, alpha, beta)
        else:
            return self.min_value(state, depth, agentNum, alpha, beta)

    def max_value(self, state, depth, agentNum, alpha, beta):
        v = -99999
        act = None
        for action in state.getLegalActions(agentNum):
            successor = state.generateSuccessor(agentNum, action)
            value, tmpAction = self.value(successor, depth, agentNum + 1, alpha, beta)
            if v < value:
                v = value
                act = action
            if v > beta: return v, act
            alpha = max(alpha, v)
        return v, action

    def min_value(self, state, depth, agentNum, alpha, beta):
        v = 99999
        act = None
        for action in state.getLegalActions(agentNum):
            successor = state.generateSuccessor(agentNum, action)
            if agentNum == state.getNumAgents() - 1:
                value, tmpAction = self.value(successor, depth + 1, 0, alpha, beta)
                if v > value:
                    v = value
                    act = action
            else:
                value, tmpAction = self.value(successor, depth, agentNum + 1, alpha, beta)
                if v > value:
                    v = value
                    act = action
            if v < alpha: return v, act
            beta = min(beta, v)
        return v, action

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 1, 0, -99999, 99999)[1]
'''

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self._getMax(gameState)[1]

    def _getMax(self, gameState, depth=0, agentIndex=0, alpha=-float('inf'),
                beta=float('inf')):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState), None
        maxVal = None
        bestAction = None
        for action in legalActions:
            value = self._getMin(gameState.generateSuccessor(agentIndex, action), depth, 1, alpha, beta)[0]
            if value is not None and (maxVal == None or value > maxVal):
                maxVal = value
                bestAction = action
            if value is not None and value > beta:
                return value, action
            if value is not None and value > alpha:
                alpha = value
        return maxVal, bestAction

    def _getMin(self, gameState, depth=0, agentIndex=0, alpha=-float('inf'),
                beta=float('inf')):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState), None
        minVal = None
        bestAction = None
        for action in legalActions:
            if agentIndex >= gameState.getNumAgents() - 1:
                value = self._getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0, alpha, beta)[0]
            else:
                value = \
                self._getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta)[0]
            if value is not None and (minVal == None or value < minVal):
                minVal = value
                bestAction = action
            if value is not None and value < alpha:
                return value, action
            if value is not None and value < beta:
                beta = value
        return minVal, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, depth, agentNum):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if depth == self.depth:
            return self.evaluationFunction(gameState), None
        if agentNum == 0:
            return self.max_value(gameState, depth, agentNum)
        else:
            return self.exp_value(gameState, depth, agentNum)

    def max_value(self, state, depth, agentNum):
        v = -99999
        act = None
        for action in state.getLegalActions(agentNum):
            successor = state.generateSuccessor(agentNum, action)
            val = self.value(successor, depth, agentNum + 1)[0]
            if v < val:
                v = val
                act = action
        return v, act

    def exp_value(self, state, depth, agentNum):
        total = 0
        length = len(state.getLegalActions(agentNum))
        for action in state.getLegalActions(agentNum):
            successor = state.generateSuccessor(agentNum, action)
            if agentNum == state.getNumAgents() - 1:
                total += self.value(successor, depth + 1, 0)[0]
            else:
                total += self.value(successor, depth, agentNum + 1)[0]
        return total / length, None
'''
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    "*** YOUR CODE HERE ***"
    if currentGameState.isWin(): return 9999999
    if currentGameState.isLose(): return -9999999
    if newFood.count() == 0: return 99999
    minFoodDistance = 99999
    newFood = newFood.asList()
    for food in newFood:
        if minFoodDistance > util.manhattanDistance(newPos, food):
            minFoodDistance = util.manhattanDistance(newPos, food)
    minGhostDistance = 99999
    GhostPos = currentGameState.getGhostPositions()
    for pos in GhostPos:
        minGhostDistance = min(minGhostDistance, util.manhattanDistance(pos, newPos))
    if minGhostDistance <= 1: return -99999
    return currentGameState.getScore() + 10 / minFoodDistance
'''
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    INF = 100000000.0
    WEIGHT_FOOD = 10.0
    WEIGHT_GHOST = -10.0
    WEIGHT_SCARED_GHOST = 100.0

    score = currentGameState.getScore()

    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(distancesToFoodList) > 0:
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        score += WEIGHT_FOOD

    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:
                score += WEIGHT_SCARED_GHOST / distance
            else:
                score += WEIGHT_GHOST / distance
        else:
            return -INF

    return score

# Abbreviation
better = betterEvaluationFunction
