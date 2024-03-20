# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import pdb
from game import Actions
from game import Directions
from util import foodGridtoDic


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    This heuristic is trivial.
    """
    return 0


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE for task1 ***"

    # comment the below line after you implement the algorithm

    class BFSSearch:
        def __init__(self, walls):
            self.walls = walls
            self.distances = {}

        def min_distance(self, source, target) -> int:
            if (source, target) in self.distances:
                return self.distances[(source, target)]
            min_distance = self.bfs_distance(source, target)
            self.distances[(source, target)] = min_distance
            return min_distance

        def get_neighbors(self, source) -> list:
            neighbors = []
            x, y = source
            if self.walls[x+1][y] is False:
                neighbors.append((x+1, y))
            if self.walls[x-1][y] is False:
                neighbors.append((x-1, y))
            if self.walls[x][y+1] is False:
                neighbors.append((x, y+1))
            if self.walls[x][y-1] is False:
                neighbors.append((x, y-1))
            return neighbors

        def bfs_distance(self, source, target) -> int:
            queue = util.Queue()
            queue.push(source)
            vistors = {}
            distances = {}
            distances[source] = 0
            while not queue.isEmpty():
                node = queue.pop()
                if node in vistors:
                    continue
                vistors[node] = True
                distance = distances[node]
                if node[0] == target[0] and node[1] == target[1]:
                    return distance
                for next in self.get_neighbors(node):
                    if next not in distances:
                        distances[next] = distance + 1
                    else:
                        distances[next] = min(distance + 1, distances[next])
                    queue.push(next)
            return None

    if 'bfs' not in problem.heuristicInfo:
        bfs = BFSSearch(problem.walls)
        problem.heuristicInfo['bfs'] = bfs
    
    bfs = problem.heuristicInfo['bfs']

    pacman_position, food_grid = state
    food_positions = food_grid.asList()

    if len(food_positions) == 0:
        return 0

    max_distance = 0
    for food_pos in food_positions:
        max_distance = max(max_distance, bfs.min_distance(pacman_position, food_pos))

    return max_distance

class MAPFProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPositions, foodGrid ) where
      pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
      foodGrid:         a Grid (see game.py) of either pacman_name or False, specifying the target food of that pacman_name. For example, foodGrid[x][y] == 'A' means pacman A's target food is at (x, y). Each pacman have exactly one target food at start
    """

    def __init__(self, startingGameState):
        "Initial function"
        "*** WARNING: DO NOT CHANGE!!! ***"
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()

    def getStartState(self):
        "Get start state"
        "*** WARNING: DO NOT CHANGE!!! ***"
        return self.start

    def isGoalState(self, state):
        "Return if the state is the goal state"
        "*** YOUR CODE HERE for task2 ***"
        # comment the below line after you implement the function
        food_grid = state[1]
        return len(food_grid.asList(False)) == (food_grid.width * food_grid.height)

    def getSuccessors(self, state):
        """
            Returns successor states, the actions they require, and a cost of 1.
            Input: search_state
            Output: a list of tuples (next_search_state, action_dict, 1)

            A search_state in this problem is a tuple consists of two dictionaries ( pacmanPositions, foodGrid ) where
              pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
              foodGrid:    a Grid (see game.py) of either pacman_name or False, specifying the target food of each pacman.

            An action_dict is {pacman_name: direction} specifying each pacman's move direction, where direction could be one of 5 possible directions in Directions (i.e. Direction.SOUTH, Direction.STOP etc)


        """
        "*** YOUR CODE HERE for task2 ***"
        # comment the below line after you implement the function

        class SuccessorActions:
            def __init__(self, walls, state):
                self.walls = walls
                self.names = []
                self.food_grid = state[1].copy()
                self.origin_position_dict = state[0]
                for name in self.origin_position_dict.keys():
                    self.names.append(name)

            def get_actions(self, postion) -> list:
                actions = []
                for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
                    dx, dy = Actions.directionToVector(direction)
                    nx, ny = int(postion[0] + dx), int(postion[1] + dy)
                    if self.walls[nx][ny] is False:
                        actions.append(((nx, ny), direction))
                return actions

            def has_actions(self, postions_dict, name, postion) -> bool:
                if name not in postions_dict:
                    return False
                return self.same_postion(postions_dict[name], postion)

            def same_postion(self, pos1, pos2) -> bool:
                return pos1[0] == pos2[0] and pos1[1] == pos2[1]
        
            def origin_postion(self, name) -> tuple:
                return self.origin_position_dict[name]

            def has_conflict(self, postions_dict : dict, next_name, next_postion) -> bool:
                for other_postion in postions_dict.values():
                    if self.same_postion(next_postion, other_postion):
                        return True
                for origin_name, origin_postion in self.origin_position_dict.items():
                    if next_name == origin_name:
                        continue
                    if self.same_postion(next_postion, origin_postion):
                        if self.has_actions(postions_dict, origin_name, self.origin_postion(next_name)):
                            return True
                return False

            def deep_copy_dict(self, source_dict) -> dict:
                dict_copy = {}
                for name, direction in source_dict.items():
                    dict_copy[name] = direction
                return dict_copy

            def get_successors(self, index : int, postions_dict : dict, direction_dict : dict, food_grid) -> list:
                successors = []
                name = self.names[index]
                now_postion = self.origin_postion(name)
                actions = self.get_actions(now_postion)
                for action in actions:
                    next_postion, direction = action
                    if self.has_conflict(postions_dict, name, next_postion):
                        continue

                    next_food_grid = food_grid.copy()
                    next_x, next_y = next_postion
                    if next_food_grid[next_x][next_y] == name:
                        next_food_grid[next_x][next_y] = False

                    next_postions_dict = self.deep_copy_dict(postions_dict)
                    next_direction_dict = self.deep_copy_dict(direction_dict)

                    next_postions_dict[name] = next_postion
                    next_direction_dict[name] = direction

                    if index + 1 == len(self.names):
                        successors.append(((next_postions_dict, next_food_grid), next_direction_dict, 1))
                    else:
                        successors = successors + self.get_successors(index + 1, next_postions_dict, next_direction_dict, next_food_grid)
                return successors

        successor = SuccessorActions(self.walls, state)
        # pdb.set_trace()
        return successor.get_successors(0, {}, {}, successor.food_grid)


def conflictBasedSearch(problem: MAPFProblem):
    """
        Conflict-based search algorithm.
        Input: MAPFProblem
        Output(IMPORTANT!!!): A dictionary stores the path for each pacman as a list {pacman_name: [action1, action2, ...]}.

    """
    "*** YOUR CODE HERE for task3 ***"

    # comment the below line after you implement the function
    util.raiseNotDefined()


"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"


class FoodSearchProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                nextFood = state[1].copy()
                nextFood[next_x][next_y] = False
                successors.append((((next_x, next_y), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class SingleFoodSearchProblem(FoodSearchProblem):
    """
    A special food search problem with only one food and can be generated by passing pacman position, food grid (only one True value in the grid) and wall grid
    """

    def __init__(self, pos, food, walls):
        self.start = (pos, food)
        self.walls = walls
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    Q = util.Queue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    Q.push(startNode)
    while not Q.isEmpty():
        node = Q.pop()
        state, cost, path = node
        if problem.isGoalState(state):
            return path
        for succ in problem.getSuccessors(state):
            succState, succAction, succCost = succ
            new_cost = cost + succCost
            newNode = (succState, new_cost, path + [succAction])
            Q.push(newNode)

    return None  # Goal not found


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, path = node
        if (not state in best_g) or (cost < best_g[state]):
            best_g[state] = cost
            if problem.isGoalState(state):
                return path
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                new_cost = cost + succCost
                newNode = (succState, new_cost, path + [succAction])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None  # Goal not found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
cbs = conflictBasedSearch
