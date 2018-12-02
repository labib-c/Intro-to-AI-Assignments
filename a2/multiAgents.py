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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("inf")*-1
        ghost = newGhostStates[0].getPosition()
        food_list = newFood.asList()
        curr_pos = currentGameState.getPacmanPosition()
        food = 100000000
        food_pos = (0, 0)
        for f in food_list:
            dist = manhattanDistance(f, newPos)
            if dist < food:
                food_pos = f
                food = dist
        score = successorGameState.getScore()
        if currentGameState.hasFood(newPos[0], newPos[1]):
            score += 200

        if manhattanDistance(newPos, ghost) < manhattanDistance(curr_pos, ghost):
            score -= 100
        else:
            score += 50

        if food < manhattanDistance(curr_pos, food_pos):
            score += 100
        else:
            score -= 50

        if newPos in currentGameState.getCapsules():
            score += 100

        if action == Directions.STOP:
            score -= 50

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def terminal(gameState, depth, count):
    return gameState.isWin() or gameState.isLose() or count == depth*gameState.getNumAgents()

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
        """
        def DFminimax(gameState, count):
            best_move = None
            if terminal(gameState, self.depth, count):
                return best_move, self.evaluationFunction(gameState)
            player = count % gameState.getNumAgents()
            if player == 0:
                best_val = float("inf")*-1
            else:
                best_val = float("inf")
            for move in gameState.getLegalActions(player):
                next_state = gameState.generateSuccessor(player, move)
                next_move, next_val = DFminimax(next_state, count+1)
                if player == 0 and next_val > best_val:
                    best_move, best_val = move, next_val
                if player != 0 and next_val < best_val:
                    best_move, best_val = move, next_val
            return best_move, best_val
        ret_move, ret_val = DFminimax(gameState, 0)
        return ret_move



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(gameState, count, alpha, beta):
            best_move = None
            if terminal(gameState, self.depth, count):
                return best_move, self.evaluationFunction(gameState)
            player = count % gameState.getNumAgents()
            if player == 0:
                best_val = float("inf")*-1
            else:
                best_val = float("inf")
            for move in gameState.getLegalActions(player):
                next_state = gameState.generateSuccessor(player, move)
                next_move, next_val = alphabeta(next_state, count+1, alpha, beta)
                if player == 0:
                    if best_val < next_val:
                        best_val, best_move = next_val, move
                    if best_val >= beta:
                        return best_move, best_val
                    alpha = max(alpha, best_val)
                if player != 0:
                    if best_val > next_val:
                        best_val, best_move = next_val, move
                    if best_val <= alpha:
                        return best_move, best_val
                    beta = min(beta, best_val)
            return best_move, best_val

        ret_move, ret_val = alphabeta(gameState, 0, -1*float("inf"), float("inf"))
        return ret_move



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
        def expectimax(gameState, count):
            best_move = None
            if terminal(gameState, self.depth, count):
                return best_move, self.evaluationFunction(gameState)
            player = count % gameState.getNumAgents()
            if player == 0:
                best_val = float("inf")*-1
            else:
                best_val = 0
            for move in gameState.getLegalActions(player):
                next_state = gameState.generateSuccessor(player, move)
                next_move, next_val = expectimax(next_state, count+1)
                if player == 0 and next_val > best_val:
                    best_move, best_val = move, next_val
                if player != 0:
                    best_val = best_val + (1.0/float(len(gameState.getLegalActions(player))))*next_val
            return best_move, best_val
        ret_move, ret_val = expectimax(gameState, 0)
        return ret_move


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    curr_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_pos = currentGameState.getGhostPositions()
    capsule_states = currentGameState.getCapsules()
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("inf") * -1
    food = []
    for f in food_list:
        dist = manhattanDistance(f, curr_pos)
        food.append(dist)
    if not food:
        food.append(0)
    ghost = []
    for g in ghost_pos:
        dist = manhattanDistance(g, curr_pos)
        ghost.append(dist)
    if not ghost:
        ghost.append(0)
    capsule = []
    for c in capsule_states:
        dist = manhattanDistance(c, curr_pos)
        capsule.append(dist)
    if not capsule:
        capsule.append(0)
    score = currentGameState.getScore()
    score -= max(food) + sum(food)/len(food)
    score += min(ghost)
    score -= max(capsule) + sum(capsule)/len(capsule)
    score -= 1000*len(food) + 500*len(capsule) - 20*len(ghost)
    return score


# Abbreviation
better = betterEvaluationFunction



