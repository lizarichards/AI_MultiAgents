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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        newFood = successorGameState.getFood().asList()
        minFoodist = float("inf")
        for food in newFood:
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        # ghost is avoided if it gets too close
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')

        return successorGameState.getScore() + 1.0/minFoodist



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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def TerminationState(self, gameState):
        # helper function to check if termination state
        # determined if is
        return gameState.isWin() or gameState.isLose()

    def Pacman(self, agentIndex):
        # helper function defining the start state
        return agentIndex == 0


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        self.miniMax(gameState)
        return self.action

    def minVal(self, gameState, agent_index, depth):
        curr_best = float("inf")
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            val = self.miniMax(successor, agent_index + 1, depth)
            curr_best = min(val, curr_best)
        return curr_best

    def maxVal(self, gameState, agent_index, depth):
        curr_best = float("-inf")
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            val = self.miniMax(successor, agent_index + 1, depth)
            curr_best = max(val, curr_best)
            if depth == 1 and curr_best == val:
                self.action = action
        return curr_best

    def miniMax(self, gameState, agent_index=0, depth=0):
        agent_index = agent_index % gameState.getNumAgents()

        if self.TerminationState(gameState):
            return self.evaluationFunction(gameState)

        if self.Pacman(agent_index):
            if depth < self.depth:
                return self.maxVal(gameState, agent_index, depth + 1)

            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minVal(gameState, agent_index, depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.miniMax(gameState)
        return self.action

    def minVal(self, gameState, agent_index, depth, alpha, beta):
        curr_best = float("inf")
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            curr_val = self.miniMax(successor, agent_index + 1, depth, alpha, beta)
            curr_best = min(curr_val, curr_best)
            if curr_best < alpha:
                return curr_best
            beta = min(beta, curr_best)
        return curr_best

    def maxVal(self, gameState, agent_index, depth, alpha, beta):
        curr_best = float("-inf")
        for action in gameState.getLegalActions(agent_index):
            succ = gameState.generateSuccessor(agent_index, action)
            curr_val = self.miniMax(succ, agent_index + 1, depth, alpha, beta)
            curr_best = max(curr_val, curr_best)
            if (depth == 1) and (curr_best == curr_val):
                self.action = action
            if curr_best > beta:
                return curr_best
            alpha = max(alpha, curr_best)
        return curr_best

    def miniMax(self, gameState, agent_index=0, depth=0,
                alpha=float("-inf"), beta=float("inf")):

        agent_index = agent_index % gameState.getNumAgents()

        if self.TerminationState(gameState):
            return self.evaluationFunction(gameState)

        if self.Pacman(agent_index):
            if depth < self.depth:
                return self.maxVal(gameState, agent_index, depth + 1, alpha, beta)
            else:
                return self.evaluationFunction(gameState)
        else:
            return self.minVal(gameState, agent_index, depth, alpha, beta)


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
        self.expectiMax(gameState)
        return self.action

    def maxVal(self, gameState, agent_index, depth):
        curr_best = float("-inf")
        for action in gameState.getLegalActions(agent_index):
            succ = gameState.generateSuccessor(agent_index, action)
            val = self.expectiMax(succ, agent_index + 1, depth)
            curr_best = max(val, curr_best)
            if depth == 1 and curr_best == val:
                self.action = action
        return curr_best

    def expectedVal(self, gameState, agent_index, depth):
        legalActions = gameState.getLegalActions(agent_index)
        curr_val = 0

        for action in legalActions:
            successor = gameState.generateSuccessor(agent_index, action)
            prob = 1.0 / len(legalActions)
            curr_val += prob * self.expectiMax(successor, agent_index + 1, depth)
        return curr_val

    def expectiMax(self, gameState, agent_index=0, depth=0):
        agent_index = agent_index % gameState.getNumAgents()

        if self.TerminationState(gameState):
            return self.evaluationFunction(gameState)

        if self.Pacman(agent_index):
            if depth < self.depth:
                return self.maxVal(gameState, agent_index, depth + 1)

            else:
                return self.evaluationFunction(gameState)
        else:
            return self.expectedVal(gameState, agent_index, depth)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    currentPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    flist = currFood.asList()
    ghostStates = currentGameState.getGhostStates()
    lowValue = -10000000.0

    currScore = currentGameState.getScore()

    for food in flist:
        distance = manhattanDistance(currentPos, food)
        currScore += (1 / float(distance))

    for ghost in ghostStates:
        gPos = ghost.getPosition()
        distance = manhattanDistance(currentPos, gPos)
        if distance == 0:
            continue
        if (distance < 3):
            currScore += 5 * (1 / float(distance))
        else:
            currScore += (1 / float(distance))

    return currScore



    # newPos = currentGameState.getPacmanPosition()
    # newFood = currentGameState.getFood().asList()
    # newGhostStates = currentGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #
    # # given by the distance from the nearest food
    # val = currentGameState.getScore()
    # food_dist = float("inf")
    # for food in newFood:
    #     foodDist = min(food_dist, util.manhattanDistance(food, newPos))
    # val += 1.0 / food_dist
    #
    # return val





# Abbreviation
better = betterEvaluationFunction
