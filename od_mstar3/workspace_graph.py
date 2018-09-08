"""
workspace_graph.py

This module defines all of the classes for the low-level graphs and
policies used in Mstar. In general terms, these classes represent:

    1.  Graphs representing the configuration space.  These graphs are
        structured so that each node in the graph represents a
        configuration, and each edge represents a permissible transition
        between two different configurations.

        *All of these graphs subclass the Graph_Interface class

    2.  Policies, which define paths in a configuration space from an
        initial configuration to a goal configuration.  Policies are
        comprised of nodes, each of which represents a configuration
        in the configuration space.  Each node in a policy has a pointer
        to its optimal neighbor, i.e., the next node in the optimal path
        to the goal node.  Policy classes compute optimal paths by using
        some search algorithm to search the graphs generated in the
        classes described above.

        *All of these graphs subclass the Policy_Interface class

There are specific implementations of policies and classes within this
module.  These are:

    1.  Grid_Graph and Grid_Graph_Conn_8: These subclass Graph_Interface
        and are used to represent simple configuration spaces in a
        2-dimensional grid.  Each point on the grid is delegated with
        either a zero or a one, to represent a free space or an
        object in that location, respectively.  Grid_Graph specifies
        a configuration space with 4 connectivity; i.e., each robot
        can only go to the space immediately above its current position,
        below its current position, or to the left or right of its
        current position.  Grid_Graph_Conn_8 specifies a configuration
        space with all the moves described in Grid_Graph, but with
        additional options options of moving diagonally.

    2.  Flood_Fill_Policy:  This subclasses Policy_Interface and
        generates an optimal path to a goal configuration by using a
        flood fill.  This method of policy generation relies on a series
        of pointers between nodes to generate a policy.  It starts
        with the goal node on an open list.  At each step, the
        algorithm pops a node off of the open list and calculates its
        neighbors, appending them to the open list.  It iterates through
        the generated neighbors and checks to see if they should point
        to the popped node, based on the popped node's cost and their
        own cost.  If they should, their pointer is changed and cost is
        updated.  Eventually, the algorithm finds the starting node, and
        an optimal policy has been generated.

        2.1 To reduce the amount of code that has to be copied each
            time a new workspace is generated, actions that deal with
            the workspace itself (rather than the configuration graph)
            are passed into Flood_Fill_Policy as functions

    3.  Astar_Policy:  This subclasses Policy_Interface and
        generates  an optimal policy to a goal configuration by using
        the A* search algorithm.  A* uses a Best-First Search approach
        to generate optimal paths in lower-order average time than flood
        fill.

        3.1 To reduce code that needs to copied for each new workspace,
            a scheme similar to that described in 2.1 has also
            been implemented in Astar.

    4.  Priority_Graph:  This subclasses Policy_Interface and
        generates an optimal policy to a goal configuration using
        an Astar_Policy graph.  However, Priority_Graph also adds
        a time slot to each coordinate.  This way, routes can be planned
        for time in addition to space.

    5.  Back_Priority_Graph:  This subclasses Priority_Graph and
        generates an optimal policy to a goal configuration. Differs
        from Priority Graph in that time dynamics are configured for
        planning backwards in time.

Finally, an Edge_Checker class is implemented in the bottom of this
module.  This class checks for collisions occurring when two robots
attempt to move past each other.

Module urrently assumes that all actions have equal cost (including
diagonal vs non-diagonal move
"""

from od_mstar3.col_set_addition import add_col_set_recursive, add_col_set
from od_mstar3.col_set_addition import NoSolutionError
import od_mstar3.SortedCollection as SortedCollection
from collections import defaultdict
from functools import wraps
#try:
#    import ipdb as pdb
#except ImportError:
#    import pdb
import od_mstar3.interface as interface
import math

# Define values delegated to free spaces and spaces with obstacles
# in the matrix of the workspace descriptor
FREE = 0
OBS = 1
# Actions for 4 connected graph
CONNECTED_4 = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
# Actions for 8 connected graph
CONNECTED_8 = ((0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
               (0, -1), (1, -1))
MAX_COST = 1000000
# DIAGONAL_COST, note that team policies imports this value as well
#DIAGONAL_COST = 2 ** .5
DIAGONAL_COST = 1.4


class wrk_node(object):

    """Holds information about a node in a policy's graph

    Defines __slots__ to decrease program memory usage by allocating
    a fixed amount of space to wrk_node object, rather than a
    dictionary holding all attributes

    Public interface of instance variables defined below:

    coord         - coordinate representing configuration
                    corresponding to this node in the configuration
                    space
    policy        - coordinate of neighboring configuration which is
                    optimal for the policy to get to the goal
                    configuration
    opt_neighbors - list of all neighbors which lead to paths that are
                    considered optimal by the policy (more than one path
                    can be optimal)
    h             - heuristic cost of configuration specified by coord
    closed, open  - specify when a policy is finalized
    iteration     - current step of policy
    """
    __slots__ = ['coord', 'policy', 'opt_neighbors', 'cost', 'h', 'closed',
                 'iteration', 'open']

    def __init__(self, coord):
        """Initialization function for nodes of astar policy graph.

        coord - coordinate of configuration which wrk_node represents
                in astar_policy graph
        """
        self.coord = coord  # Want to store as tuples
        self.policy = None  # Holds coordinate of next neighbor to visit
        # Holds all optimal neighbors, intended to make replanning the
        # policy to find an optimal, collision avoiding path easier
        self.opt_neighbors = []  # currently only generated by _road_rules,
        # also used to store neighbor offsets for EPEA*
        self.cost = MAX_COST  # Cost to goal
        # Used for extension easier to running resumable A* search
        self.h = 0
        # Used to determine when a policy is finalized
        self.closed = False
        self.iteration = -1
        self.open = False


# Simple memoization decorator, can be used for any function
# Although no code in this module has been effectively sped up with this
# decorator yet, this will hopefully be useful with more complex graphs
# and configuration spaces in the future
def memoize(f):
    memo = {}

    @wraps(f)
    def inner(*args, **kwargs):
        try:
            return memo[args]
        except KeyError:
            z = memo[args] = f(*args)
            return z
    return inner


def node_cmp(n1, n2):
    """ Sort nodes by cost """
    if n1.cost < n2.cost:
        return -1
    elif n1.cost > n2.cost:
        return 1
    # Returning 0 allows for stable sorting, i.e. equal objects stay in
    # the same order, which should provide a bit of a performance boost,
    # as well as a bit of consistency
    return 0


class Networkx_DiGraph(interface.Graph_Interface):
    """Simple wrapper for networkx graphs, in particular, supports
    digraphs.

    Requires a modified policy which can account for DiGraphs, because
    the forward and backword neighbors are not the same thing
    """

    def __init__(self, graph):
        """graph - networkx.DiGraph specifying the configuration space.
                   assumes, cost is stored in the cost parameter
        """
        self.graph = graph

    def get_edge_cost(self, coord1, coord2):
        """Returns edge_cost of going from coord1 to coord2

        coord1, coord2 - node identification

        returns:
        edge cost
        """
        return self.graph[coord1][coord2]['cost']

    def get_neighbors(self, coord):
        """Returns the out-neighbors of the specified node

        coord - identifier of the node to query

        returns:
        list of node identifiers of neighboring nodes
        """
        return self.graph.neighbors(coord)

    def get_in_neighbors(self, coord):
        """Returns the in-neighbors of the specified node

        coord - identifier of the node to query

        returns:
        list of node identifiers of in-neighbors
        """
        return [c[0] for c in self.graph.in_edges(coord)]


class Grid_Graph(interface.Graph_Interface):
    """ Represents configuration space for grid workspace

    This graph serves to generate the configuration graph for a
    gridded workspace.  This workspace must be 4-connected, so
    that a robot can go to the grid spaces located one space
    up, one space to the left, one space to the right, and one
    space down from its current coordinate.
    """

    def __init__(self, world_descriptor, diagonal_cost=False):
        """Initialization for grid graph

        world_descriptor - Rectangular matrix, 0 for free cell, 1 for obstacle
        diagonal_cost    - Boolean, apply 2**.5 for diagonal cost
        """
        self.world_descriptor = world_descriptor
        self.width = len(world_descriptor)
        self.height = len(world_descriptor[0])
        self.actions = CONNECTED_4
        if diagonal_cost:
            self._diagonal_cost = DIAGONAL_COST
        else:
            self._diagonal_cost = 0

    def get_edge_cost(self, coord1, coord2):
        """Retrieves config edge cost between two configurations

        Grid_Graph has a fixed edge cost of one, effectively optimizing
        make-span

        coord1 - coordinate of source vertex
        coord2 - coordinate of target vertex

        Returns edge_cost of going from coord1 to coord2.
        """
        if (self._diagonal_cost and coord1[0] != coord2[0] and
                coord1[1] != coord2[1]):
            return self._diagonal_cost
        return 1

    def get_neighbors(self, coord):
        """Returns collision free neighbors of the specified coordinate.

        coord - (x, y) coordinate of the node for which neighbors are
                being generated

        Return value in form of list of (x, y) tuples giving coordinates
        of neighbors, including self
        """
        neighbors = []
        min_cost = MAX_COST
        for i in self.actions:
            new_coord = (i[0] + coord[0], i[1] + coord[1])
            # check if points to a coordinate in the graph
            if (new_coord[0] < 0 or new_coord[0] >= self.width or
                    new_coord[1] < 0 or new_coord[1] >= self.height):
                continue
            if self.world_descriptor[new_coord[0]][new_coord[1]] == OBS:
                # Points to obstacle
                continue
            # Valid single robot action
            neighbors.append(new_coord)

        return neighbors

    def get_in_neighbors(self, coord):
        """Returns the collision free in-neighbors of the specified
        coordinate.

        Equivalent to get_neighbors, because the graph is undirected

        coord - (x, y) coordinate of vertex for which to return the
                in-neighbors

        Returns:
        List of coordinates of in-neighbors
        """
        return self.get_neighbors(coord)


class Grid_Graph_Conn_8(Grid_Graph):
    """ Configuration graph for gridded workspace with 8 connection

    This graph serves to generate the configuration graph for gridded
    workspace where each point in the grid has eight neighbors.
    """

    def __init__(self, world_descriptor, diagonal_cost=False):
        """Initialization for grid graph with 8 connectivity

        world_descriptor    - Rectangular matrix, 0 for free cell, 1 for
                              obstacle
        """
        super(Grid_Graph_Conn_8, self).__init__(world_descriptor,
                                                diagonal_cost=diagonal_cost)
        self.actions = CONNECTED_8


class GridGraphConn4WaitAtGoal(Grid_Graph):
    """Variant of workspace_graph.Grid_Graph that allows for the robot
    to wait at its goal with reduced cost
    Note: this can not be used directly for CMS to allow reduced waiting
    cost when a team is not ready to be formed, as cost should not be
    reduced when a team is ready to be formed.
    """

    def __init__(self, world_descriptor, goal, wait_cost=.0,
                 diagonal_cost=False):
        """Initialization for grid graph

        world_descriptor - Rectangular matrix, 0 for free cell, 1 for
                           obstacle
        goal             - goal of the robot
        wait_cost        - cost to incur for waiting at the goal
                           configuration
        diagonal_cost    - incur DIAGONAL_COST for moving diagonally,
                           1 otherwise. included to support subclasses
        """
        super(GridGraphConn4WaitAtGoal, self).__init__(
            world_descriptor, diagonal_cost=diagonal_cost)
        self._goal = goal
        self._wait_cost = wait_cost

    def get_edge_cost(self, coord1, coord2):
        """Retrieves edge cost between two configurations

        Waiting at the goal incurs cost wait_cost, while any other
        action incurs cost self._wait_cost

        coord1 - coordinate of source vertex
        coord2 - coordinate of target vertex

        Returns edge_cost of going from coord1 to coord2.
        """
        if coord1 == self._goal and coord2 == self._goal:
            return self._wait_cost
        return super(GridGraphConn4WaitAtGoal, self).get_edge_cost(coord1,
                                                                   coord2)


class GridGraphConn8WaitAtGoal(GridGraphConn4WaitAtGoal):
    """Variant of workspace_graph.Grid_Graph__con_8 that allows for the
    robot to wait at its goal with reduced cost
    Note: this can not be used directly for CMS to allow reduced waiting
    cost when a team is not ready to be formed, as cost should not be
    reduced when a team is ready to be formed.
    """

    def __init__(self, world_descriptor, goal, wait_cost=.0,
                 diagonal_cost=False):
        """Initialization for grid graph

        world_descriptor - Rectangular matrix, 0 for free cell, 1 for
                           obstacle
        goal             - goal of the robot
        wait_cost        - cost to incur for waiting at the goal
                           configuration
        diagonal_cost    - incur DIAGONAL_COST for moving diagonally if True,
                           incur 1 if False
        """
        super(GridGraphConn8WaitAtGoal, self).__init__(
            world_descriptor, goal, wait_cost=wait_cost,
            diagonal_cost=diagonal_cost)
        self.actions = CONNECTED_8


def Workspace_Graph(world_descriptor, goal=None, connect_8=False,
                    road_rules=True):
    """Wrapper function for returning Flood_Fill_Policy objects

    Function returns objects with different args depending on the
    connect_8 flag

    world_descriptor - two-dimensional matrix representing the space in
                       which the robot can travel.  A value of 1 in the
                       space represents an obstacle, and a value of 0
                       represents an open space
    goal             - position [x,y] of the goal of the policy
    connect_8        - boolean determining whether Grid_Graph or
                       Grid_Graph_Conn_8 is used
    road_rules       - boolean supplied to policy object to determine if
                       rightmost neighbor node should always be used
    """
    if connect_8:
        return Flood_Fill_Policy(world_descriptor, Grid_Graph_Conn_8,
                                 goal, road_rules)
    return Flood_Fill_Policy(world_descriptor, Grid_Graph, goal,
                             road_rules)


def compute_heuristic_conn_8(init_pos, coord):
    """Returns a heuristic for distance between coord and init_pos

    init_pos - coordinate of position of goal configuration
    coord    - coordinate of configuration for which heuristic is
               being computed

    Returns the heuristic distance to goal
    """
    return max(map(lambda x, y: abs(x - y), coord, init_pos))


def compute_heuristic_conn_8_diagonal(init_pos, coord):
    """Returns a heuristic for distance between coord and init_pos

    Used when moving diagonally costs DIAGONAL_COST instead of 1

    init_pos - coordinate of position of goal configuration
    coord    - coordinate of configuration for which heuristic is
               being computed

    Returns the heuristic distance to goal
    """
    x_diff = abs(init_pos[0] - coord[0])
    y_diff = abs(init_pos[1] - coord[1])
    min_dist = min(x_diff, y_diff)
    max_dist = max(x_diff, y_diff)
    return DIAGONAL_COST * min_dist + (max_dist - min_dist)


def compute_heuristic_conn_4(init_pos, coord):
    """Returns Manhattan heuristic for distance from coord to init_pos

    init_pos - coordinate of position of goal configuration
    coord    - coordinate of configuration for which heursitic is
               being computed

    Returns the heuristic distance to goal through a
    Manhattan metric calculation.
    """
    return sum(map(lambda x, y: abs(x - y), coord, init_pos))


def Astar_Graph(world_descriptor, goal=None, connect_8=False,
                diagonal_cost=False, makespan=False):
    """Wrapper function for returning Astar_Policy objects

    Different heuristic functions are given to Astar_Policy object
    depending on whether the gridworld is 8 connected or not

    world_descriptor - two-dimensional matrix which describes the
                       gridworld with obstacles. Each point in the
                       matrix is either a zero (no obstacle) or a
                       one (obstacle)
    goal             - position (x, y) of the goal of the policy
    connect_8        - boolean determining whether each coordinate
                       in the gridworld has eight neighbors
                       (including all diagonal neighbors) or only
                       four (cardinal neighbors)
    diagonal_cost    - boolean, apply DIAGONAL_COST for diagonal costs if True,
                       apply 1 if False
    makespan         - minimize makespan instead of minimizing time
    """
    if makespan:
        if connect_8:
            if diagonal_cost:
                h_func = compute_heuristic_conn_8_diagonal
            else:
                h_func = compute_heuristic_conn_8
            return Astar_Policy(
                world_descriptor,
                lambda x: Grid_Graph_Conn_8(x, diagonal_cost=diagonal_cost),
                goal=goal, compute_heuristic=h_func)
        else:
            return Astar_Policy(world_descriptor, Grid_Graph, goal=goal,
                                compute_heuristic=compute_heuristic_conn_4)
    if connect_8:
        if diagonal_cost:
            h_func = compute_heuristic_conn_8_diagonal
        else:
            h_func = compute_heuristic_conn_8
        return Astar_Policy(
            world_descriptor,
            lambda x: GridGraphConn8WaitAtGoal(x, goal,
                                               wait_cost=1.0,
                                               diagonal_cost=diagonal_cost,
                                               ),
            goal, h_func)
    return Astar_Policy(world_descriptor,
                        lambda x: GridGraphConn4WaitAtGoal(
                            x, goal, wait_cost=1.0,
                            diagonal_cost=diagonal_cost),
                        goal, compute_heuristic_conn_4)


class Astar_Policy(interface.Policy_Interface):

    """Class that implements Astar to search config space

    Uses resumable A* search instead of the flood fill used in
    workspace graph, as the optimal policy computation is dominating
    the time required for rM*  when inflated.

    To avoid copying large amounts of code for each new workspace,
    all functions interacting with the workspace are passed into this
    class as arguments.
    """
    def __init__(self, world_descriptor, config_graph, goal=None,
                 compute_heuristic=compute_heuristic_conn_4):
        """Initialization function for Astar_Policy

        world_descriptor  - two-dimensional matrix which describes the
                            gridworld with obstacles. Each point in the
                            matrix is either a zero (no obstacle) or a
                            one (obstacle)
        config_graph      - a callable that takes a single argument, the
                            world descriptor, and returns an object that
                            represents the configuration graph, which
                            implements the methods defined by
                            Graph_Interface
        goal              - (x, y)  target, optional, if not supplied,
                            will not generate policy
        compute_heuristic - helper function used to calculate the
                            heuristic distance to the goal. Passed in
                            because it interacts with the workspace
        """
        self.cspace = config_graph(world_descriptor)
        self.graph = {}
        self.iteration = 0
        self.goal = goal
        self.init_pos = self.goal
        self.compute_heuristic = compute_heuristic
        self.goal_node = self._get_node(self.goal)
        # We implicitly assume a self loop by setting the goal node's
        # policy to be its own coordin
        self.goal_node.policy = self.goal_node.coord
        self.goal_node.cost = 0
        self.goal_node.open = True
        self.open_list = SortedCollection.SortedCollection(
            [self.goal_node], key=lambda x: -x.cost - x.h)

    def _get_node(self, coord):
        """Returns node specified by coord

        In addition, updates its heursitic and iteration values.  If no
        such node exists, it is created.

        coord - coordinate of node to return
        """
        try:
            node = self.graph[coord]
        except KeyError:
            node = self.graph[coord] = wrk_node(coord)

        if self.iteration > node.iteration:
            node.iteration = self.iteration
            node.h = self.compute_heuristic(self.init_pos, coord)
        return node

    def _compute_path(self, coord):
        """Extends the search to reach the specified node

        coord - (x,y) coordinate of targeted configuration

        Tries to compute path from coord to goal.  If successful,
        returns next coordinate in path to goal from coord.  If not
        successful, raises an NoSolutionError.
        """
        if self.init_pos == self.goal:
            self.init_pos = coord
            # First need to update the heuristic for nodes in the open
            # list
        # Only change the heuristic for the intial coordinate, when the
        # open list is empty, so don't actually have to resort the open
        # list
        # Open list may be empty if trying after trying to find paths to
        # two unreachable nodes.  This will only be done my
        # multi_assignment_mstar while trying to compute the assignment
        # cost matrix.  Besides which, this will trigger a
        # NoSolutionError in case such a situtation is not supposed to
        # be found assert len(self.open_list) > 0
        while len(self.open_list) > 0:
            node = self.open_list.pop()
            if node.closed:
                continue
            node.closed = True
            node.open = False
            # Need to add the neighbors before checking if this is the
            # goal, so search can be resumed without being blocked by
            # this position
            neighbors = self.get_neighbors(node.coord)
            for i in neighbors:
                tnode = self._get_node(i)
                if (tnode.closed or tnode.cost <= node.cost +
                        self.get_edge_cost(i, node.coord)):
                    continue
                tnode.cost = node.cost + self.get_edge_cost(
                    i, node.coord)
                tnode.policy = node.coord
                tnode.open = True
                # Can add tnode directly, and will just skip any
                # inconsistent copies
                self.open_list.insert_right(tnode)
            if node.coord == coord:
                # Done, so return the next step
                return node.policy
        raise NoSolutionError('Couldn\'t finish individual policy')

    def get_step(self, coord):
        """Gets the policy for the given coordinate

        If no policy exists, extends planning to reach the coordinate

        coord - (x, y) configuration

        Returns a coordinate of the next node in the policy
        """
        node = self._get_node(coord)
        if node.closed:
            # Have already computed the optimal policy here
            return node.policy
        self.iteration += 1
        try:
            return self._compute_path(coord)
        except NoSolutionError:
            # Couldn't find a path to goal, so return None
            return None

    def get_cost(self, coord):
        """Returns the cost of moving from given position to goal

        Cost is for moving from coordinate specified at coord
        to the goal configuration.

        coord - (x, y) configuration
        """
        node = self._get_node(coord)
        if node.closed:
            return node.cost
        self.iteration += 1
        self._compute_path(coord)
        assert node.closed
        return node.cost

    def get_edge_cost(self, coord1, coord2):
        """Returns cost of config transition from coord1 to coord2

        Wrapper function for returning the config space's
        get_edge_cost from coord1 to coord2

        coord1 - initial coordinate in transition
        coord2 - final coordinate in transition

        returns:
        edge cost of going from coord1 to coord2
        """
        return self.cspace.get_edge_cost(coord1, coord2)

    def _gen_limited_offset_neighbors(self, coord):
        """Stores the neighbors of a node by changes in f-value

        f-value - the sum of cost to reach and cost to go.

        coord - (x, y) configuration for which limited offset neighbors
                are generated
        """
        # Repurposing a preexisting field, so need to change to a
        # defaultdict
        node = self._get_node(coord)
        node.opt_neighbors = defaultdict(lambda: [])
        base_cost = self.get_cost(coord)
        # Need to compute offsets
        for neib in self.get_neighbors(coord):
            # difference in path cost using different paths, need to
            # handle staying at the goal seperately
            if neib == self.goal and neib == coord:
                offset = 0
            else:
                offset = self.get_cost(neib) - base_cost + 1
            node.opt_neighbors[offset].append((offset, neib))
        node.opt_neighbors = dict(node.opt_neighbors)

    def get_limited_offset_neighbors(self, coord, max_offset, min_offset=0):
        """Returns set of neighbors specified by the offsets

        More specifically, returns the set of neighbors for which the
        maximum difference in path cost if passed through is less than
        the specified value.

        (i.e. if you are forced to pass through coordinate x, instead of
        the optimal step, what is the difference in cost)?

        coord - coordinates of the node to find neighbors of
        max_offset - the maximum increase in path cost to encur in
                     choice of neighbors
        min_offset - minimum increae in path cost to encur in a neighbor

        returns:
        a list of tuples of the form (offset, coordinate)
        """
        node = self._get_node(coord)
        if not node.opt_neighbors:
            self._gen_limited_offset_neighbors(coord)
        # Have already pre-computed the results
        out = []
        for offset, neighbors in node.opt_neighbors.iteritems():
            if offset < min_offset:
                continue
            if offset > max_offset:
                return out
            out.extend(neighbors)
        return out

    def get_offset_neighbors(self, coord, offset):
        """Generates offset neighbors for node specified by coord

        If no offset neighbors exist, they are created

        Only offset neighbors at a certain offset are returned

        coord  - (x,y) configuration for which neighbors are being
                 generated
        offset - value of offset determing which neighbors are
                 included in return value

        returns:
        list of tuples of form (offset, neighbor)
        """
        node = self._get_node(coord)
        if not node.opt_neighbors:
            self._gen_limited_offset_neighbors(coord)
        return node.opt_neighbors[offset]

    def get_offsets(self, coord):
        """Return the possible offsets of the neighbors.

        The offset of a neighbor is the difference in the cost of the
        optimal path from coord to the cost of the best path constrained
        to pass through a specific neighbor.  Used in EPEA*

        coord - (x,y) configuration for which neighbors are being
                generated and their offsets returned

        Returns list of offsets of all neighbor nodes to coord
        """
        node = self._get_node(coord)
        if not node.opt_neighbors:
            self._gen_limited_offset_neighbors(coord)
        return node.opt_neighbors.keys()

    def get_neighbors(self, coord, opt=False):
        """Wrapper function for get_neighbors function of underlying
        config_space graph.

        opt - only optimal neighbors are returned
        coord - configuration for which neighbors are being returned

        Returns list of tuples, where each tuple is a coordinate
        """
        neighbors = self.cspace.get_neighbors(coord)
        if not opt:
            return neighbors
        for i in neighbors:
            if opt:
                cost = self.get_cost(i)
                if cost < min_cost:
                    min_cost = cost
        opt_neighbors = []
        for i in neighbors:
            if self.get_cost(i) == min_cost:
                opt_neighbors.append(i)
        return opt_neighbors

    def get_graph_size(self, correct_for_size=True):
        """Gets the size of the graph

        correct_for_size - just intended to match signatures

        Returns the number of nodes used for this graph
        """
        return sum(map(len, self.graph))


class Astar_DiGraph_Policy(Astar_Policy):

    """Class that implements Astar to search configuration spaces that
    are represented as a di graph

    Differs slightly from Astar_Policy in using the get_in_neighbors
    function when computing a policy, to explicitly plan back in time

    Uses resumable A* search instead of the flood fill used in
    workspace graph, as the optimal policy computation is dominating
    the time required for rM*  when inflated.

    To avoid copying large amounts of code for each new workspace,
    all functions interacting with the workspace are passed into this
    class as arguments.
    """
    def __init__(self, world_descriptor, config_graph, goal=None,
                 compute_heuristic=compute_heuristic_conn_4):
        """Initialization function for Astar_Policy

        world_descriptor  - two-dimensional matrix which describes the
                            gridworld with obstacles. Each point in the
                            matrix is either a zero (no obstacle) or a
                            one (obstacle)
        config_graph      - a class which is used to represent the
                            config space of the robot
        goal              - (x, y)  target, optional, if not supplied,
                            will not generate policy
        compute_heuristic - helper function used to calculate the
                            heuristic distance to the goal. Passed in
                            because it interacts with the workspace
        """
        super(Astar_DiGraph_Policy, self).__init__(
            world_descriptor, config_graph, goal=goal,
            compute_heuristic=compute_heuristic)

    def _compute_path(self, coord):
        """Extends the search to reach the specified node

        Explicitly plans in reverse from the goal to the target, using
        get_in_neighbors to compute node expansion, instead of
        Astar_Graph, which uses the get_neighbors function.

        coord - (x,y) coordinate of targeted configuration

        Tries to compute path from coord to goal.  If successful,
        returns next coordinate in path to goal from coord.  If not
        successful, raises an NoSolutionError.
        """
        if self.init_pos == self.goal:
            self.init_pos = coord
            # First need to update the heuristic for nodes in the open
            # list
        # Only change the heuristic for the intial coordinate, when the
        # open list is empty, so don't actually have to resort the open
        # list
        # Open list may be empty if trying after trying to find paths to
        # two unreachable nodes.  This will only be done my
        # multi_assignment_mstar while trying to compute the assignment
        # cost matrix.  Besides which, this will trigger a
        # NoSolutionError in case such a situtation is not supposed to
        # be found assert len(self.open_list) > 0
        while len(self.open_list) > 0:
            node = self.open_list.pop()
            if node.closed:
                continue
            node.closed = True
            node.open = False
            # Need to add the neighbors before checking if this is the
            # goal, so search can be resumed without being blocked by
            # this position
            neighbors = self.get_in_neighbors(node.coord)
            for i in neighbors:
                tnode = self._get_node(i)
                if (tnode.closed or tnode.cost <= node.cost +
                        self.get_edge_cost(i, node.coord)):
                    continue
                tnode.cost = node.cost + self.get_edge_cost(
                    i, node.coord)
                tnode.policy = node.coord
                tnode.open = True
                # Can add tnode directly, and will just skip any
                # inconsistent copies
                self.open_list.insert_right(tnode)
            if node.coord == coord:
                # Done, so return the next step
                return node.policy
        raise NoSolutionError('Couldn\'t finish individual policy')

    def get_in_neighbors(self, coord):
        """Wraper for the get_in_neighbors function of the underlying
        config_space graph

        coord - coordinate of whom the predecessors (in neighbors) are
                returned

        returns:
        list of coordinates of the predecessors of coord
        """
        return self.cspace.get_in_neighbors(coord)


class Priority_Graph(interface.Policy_Interface):
    """Simple wrapper for A* graph that uses priority planning.

    Adds/removes a time coordinate to allow for priority planning.
    Implemented this way to make Indpendence_Detection happier, as it
    makes use both of basic Astar_Policy and priority planners of
    various forms. This way, any work done by the Astar_Policy can be
    leveraged for the priority planner, and vice versa
    """
    def __init__(self, astar_policy, max_t=None):
        """initialization for Priority_Graph

        astar_policy       - the graph to wrap

        max_t - greatest t - value allowed
        """
        self.astar_policy = astar_policy
        self.max_t = max_t

    def get_step(self, coord):
        """Gets the policy for the given coordinate,

        If necessary, extends planning to reach said coordinate.  Will
        increment time by 1

        coord - (x, y, t) position and time coordinate for the specified
                node
        """
        # Can do this by stripping time, querrying the underlying
        # astar_policy, then appending the appropriate new time
        t = coord[-1] + 1
        # Check if this would exceed maximal value
        if self.max_t is not None:
            t = min(self.max_t, t)
        step = self.astar_policy.get_step(coord[:2])
        return step + (t, )

    def get_cost(self, coord):
        """Gets cost of moving to goal from coord

        coord - (x, y, t)  coordinates of node for which to get cost

        Returns cost of moving from the given position to goal
        """
        return self.astar_policy.get_cost(coord[:2])

    def set_max_t(self, max_t):
        """Sets the maximum time value the graph will use.

        Allows for easy changes for different constraints
        """
        self.max_t = max_t

    def get_neighbors(self, coord):
        """Gets the neighbors of the specified space-time point

        coord - coordinate of configuration for which neighbors are
                being returned

        Returns neighbors of coord in config space, with a time stamp
        one greater than that of coord
        """
        pos_neighbors = self.astar_policy.get_neighbors((coord[0], coord[1]))
        return map(lambda x: (x[0], x[1], min(self.max_t, coord[-1] + 1)),
                   pos_neighbors)


class Back_Priority_Graph(Priority_Graph):

    """Simple wrapper for A* graph which just adds/removes a time
    coordinate to allow for priority planning.

    Implemented this way to make Indpendence_Detection happier, as it
    makes use both of basic Astar_Policy and priority planners of various
    forms. This way, any work done by the Astar_Policy can be leveraged
    for the priority planner, and vice versa.

    Differs from Priority Graph in that time dynamics are configured for
    planning backwards in time.  Need to query max_t in each instance,
    as multiple Constrained_Planners will make use of a single
    Back_Priority Graph, and no other planner should be using one
    """

    def __init__(self, astar_policy, max_t=None, prune_paths=True):
        """
        astar_policy - the graph to wrap
        max_t       - greatest t-value allowed
        prune_paths - whether to prune neighbors that cannot reach the
                      goal of astar_policy within the time specified.
                      This is the default behavior.  Disabling when
                      running task swapping allows for paths to be found
                      to multiple initial configurations
        """
        Priority_Graph.__init__(self, astar_policy, max_t=max_t)
        self.prune_paths = prune_paths

    def get_neighbors(self, coord, max_t):
        """Gets the neighbors of the specified space-time point"""
        self.max_t = max_t
        if coord[-1] == 0 and self.max_t != 0:
            return []
        pos_neighbors = self.astar_policy.get_neighbors((coord[0], coord[1]))
        if coord[-1] == self.max_t:
            neighbors = []
            for pos in pos_neighbors:
                neighbors.append((pos[0], pos[1], self.max_t))
                # Make sure that you can actually get form the initial
                # position to the suggested vertex in time
                if self.prune_paths:
                    if (not self.max_t == 0 and
                            self.astar_policy.get_cost(pos) <= coord[-1] - 1):
                        neighbors.append((pos[0], pos[1], coord[-1] - 1))
                else:
                    # Don't check on whether there is time to reach the
                    # intial configuration
                    neighbors.append((pos[0], pos[1], coord[-1] - 1))
            return neighbors
        if self.prune_paths:
            return [(x[0], x[1], coord[-1] - 1) for x in pos_neighbors
                    if self.astar_policy.get_cost(x) <= coord[-1] - 1]
        else:
            return [(x[0], x[1], coord[-1] - 1) for x in pos_neighbors]

    def get_forwards_neighbors(self, coord, max_t):
        """Gets the forward time dynamics neighbors of this point"""
        self.max_t = max_t
        return Priority_Graph.get_neighbors(self, coord)

    def get_cost(self, coord, max_t):
        """Returns the cost of moving from given position to goal

        coord - (x, y, t)  coordinates of node for which to get cost

        """
        self.max_t = max_t
        return Priority_Graph.get_cost(self, coord)

    def get_step(self, coord, max_t):
        """Gets the policy for the given coordinate, extending planning
        to reach said coordinate if necessary.  Will increment time by 1

        coord - (x, y, t) position and time coordinate for the specified
                node

        """
        self.max_t = max_t
        return Priority_Graph.get_step(self, coord)


class Limited_Astar_Policy(Astar_Policy):
    """Uses resumable A* search instead of the flood fill used in
    workspace graph, as the optimal policy computation is dominating the
    time required for rM* when inflated.

    Also takes a networkx graph, called limit graph, which specifies the
    legal edges

    """
    def __init__(self, world_descriptor, goal, limit_graph, connect_8=False):
        Astar_Policy.__init__(self, world_descriptor, goal, connect_8)
        self.limit_graph = limit_graph

    def get_neighbors(self, coord):
        """Returns the neighbors of the given coordinate in the limit
        graph

        """
        return self.limit_graph.neighbors(coord)


class Edge_Checker(interface.Planner_Edge_Checker):
    """Used to wrap edge checking so more complex graphs can be cleanly
    handled (may require keeping track of state for non-trivial graphs

    """
    def __init__(self):
        """Takes no arguments, because on grid graph, only the
        coordinates matter

        """
        pass

    def simple_pass_through(self, c1, c2):
        """Simply check for collisions, avoid the additional overhead

        for use with basic OD (op_decomp)

        c1 - coordinate at time t
        c2 - coordinate at time t + 1

        returns:
        True if pass through collision, else false

        """
        for i in range(len(c1)):
            for j in range(i + 1, len(c1)):
                if c1[i] == c2[j] and c1[j] == c2[i]:
                    return True
        return False

    def simple_col_check(self, c1):
        """Checks for robot-robot collisions at c1,

        for use with basic OD (op_decomp)

        returns:
        True if collision exists

        """
        for i in range(len(c1)):
            for j in range(i + 1, len(c1)):
                if c1[i] == c1[j]:
                    return True
        return False

    def simple_cross_over(self, c1, c2):
        """Check for cross over collisions in 8-connected worlds

        returns:
        True if collision is detected

        """
        for i in range(len(c1)):
            for j in range(i + 1, len(c1)):
                # compute displacement vector
                disp = [c1[i][0] - c1[j][0], c1[i][1] - c1[j][1]]
                if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                    continue
                # compute previous? displacement vector.  Have a pass
                # through or cross over collision if the displacement
                # vector is the opposite
                if (disp[0] == -(c2[i][0] - c2[j][0]) and
                        disp[1] == -(c2[i][1] - c2[j][1])):
                    return True
        return False

    def simple_incremental_cross_over(self, c1, c2):
        """Check for cross over collisions in 8-connected worlds.

        Assumes that collision checking has been performed for everything
        but the last robot in the coordinates.  To be used to save a bit
        of time for partial expansion approaches

        """
        for i in range(len(c1) - 1):
            disp = [c1[i][0] - c1[-1][0], c1[i][1] - c1[-1][1]]
            if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                continue
            # compute previous? displacement vector.  Have a pass through
            # or cross over collision if the displacement vector is the
            # opposite
            if (disp[0] == -(c2[i][0] - c2[-1][0]) and
                    disp[1] == -(c2[i][1] - c2[-1][1])):
                return True
        return False

    def simple_incremental_col_check(self, c1):
        """Checks for robot-robot collisions at c1,

        for use with basic OD (op_decomp)

        returns:
        True if collision exists

        """
        for i in range(len(c1) - 1):
            if c1[i] == c1[-1]:
                return True
        return False

    def single_bot_outpath_check(self, cur_coord, prev_coord, cur_t, paths):
        """Tests for collisions when moving from prev_coord to cur_coord
        with the robots in paths.

        cur_coord - position of a single robot

        Returns:

        True if a collision is found,
        False otherwise

        """
        if paths is None:
            return False
        prev_t = cur_t - 1
        check_t = min(cur_t, len(paths) - 1)
        new_cols = 0
        for bot in range(len(paths[0])):
            # Check for simultaneous occupation
            if (cur_coord[0] == paths[check_t][bot][0] and
                    cur_coord[1] == paths[check_t][bot][1]):
                return True
            if cur_t >= len(paths):
                # Can't have edge collisions when out-group robots
                # aren't moving
                continue
            # Check for pass-through/cross over collisions
            disp = [prev_coord[0] - paths[prev_t][bot][0],
                    prev_coord[1] - paths[prev_t][bot][1]]
            if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                continue
            # Compute current displacement vector, and check for
            # inversion
            if (disp[0] == -(cur_coord[0] - paths[cur_t][bot][0]) and
                    disp[1] == -(cur_coord[1] - paths[cur_t][bot][1])):
                return True
        return False

    def simple_prio_col_check(self, coord, t, paths, pcoord=None,
                              conn_8=False):
        """Returns true, if collision is detected, false otherwise
        at the moment only used to check the obstacle collisions, but
        didn't want to reject the other code already

        coord - coord of potential new neighbor
        t - current time step
        paths - previously found paths
        pcoord - previous coordinate of the path

        """
        if not isinstance(coord, tuple):
            coord = tuple(coord)
        if paths is not None:
            t = min(t, len(paths) - 1)
            # only one path
            if isinstance(paths[0][0], int):
                paths = map(lambda x: [x], paths)
            for bot in range(len(paths[t])):
                if not isinstance(paths[t][bot], tuple):
                    paths[t][bot] = tuple(paths[t][bot])
                # (a) simultaneous occupation of one node
                if coord == paths[t][bot]:
                        return True
                # (b) pass through and cross over collision
                if pcoord is not None:
                    if not isinstance(pcoord, tuple):
                        pcoord = tuple(pcoord)
                if not isinstance(paths[t - 1][bot], tuple):
                    paths[t - 1][bot] = tuple(paths[t - 1][bot])
                if paths[t - 1][bot] == coord and paths[t][bot] == pcoord:
                    return True
                # (c) cross over collision in case of conn_8
                if conn_8:
                    if self.single_bot_cross_over(paths[t][bot],
                                                  paths[t - 1][bot], coord,
                                                  pcoord):
                        return True
        # No collision
        return False

    def col_check(self, c1, recursive):
        """Checks for collisions at a single point.  Returns either a M*
        or rM* collision set in the form of sets, depending on the
        setting of recursive.

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                if c1[i] == c1[j]:
                    col_set = adder([frozenset([i, j])], col_set)
        return col_set

    def incremental_col_check(self, c1, recursive):
        """Checks for collisions at a single point.  Returns either a M*
        or rM* collision set in the form of sets, depending on the
        setting of recursive.  Only checks whether the last robot is
        involved in a collision, for use with incremental methods

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        j = len(c1) - 1
        for i in range(len(c1) - 1):
            if c1[i] == c1[j]:
                col_set = adder([frozenset((i, j))], col_set)
        return col_set

    def cross_over(self, c1, c2, recursive=False):
        """Detects cross over collisions as well as pass through
        collisions

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                # compute current displacement vector
                if c1[i] is None or c1[j] is None or c2[i] is None or c2[j] \
                        is None:
                    continue
                disp = (c1[i][0] - c1[j][0], c1[i][1] - c1[j][1])
                if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                    continue
                # Compute previous displacement vector.  Have a cross over or
                # pass through collision if the two displacement vectors are
                # opposites
                # pdisp = [c2[i][0] - c2[j][0], c2[i][1] - c2[j][1]]
                if (disp[0] == -(c2[i][0] - c2[j][0]) and
                        disp[1] == -(c2[i][1] - c2[j][1])):
                    col_set = adder([frozenset([i, j])], col_set)
        return col_set

    def incremental_cross_over(self, c1, c2, recursive=False):
        """Detects cross over collisions as well as pass through
        collisions.

        Only checks if the last robot is involved in a collision, for use
        with partial expansion approaches.

        c1 - the initial configuration.
        c2 - the final configuration. c1 may include additional robots,
             if necessary

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        j = len(c2) - 1
        for i in range(len(c2) - 1):
            # compute current displacement vector
            disp = (c1[i][0] - c1[j][0], c1[i][1] - c1[j][1])
            if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                continue
            # Compute previous displacement vector.  Have a cross over or
            # pass through collision if the two displacement vectors are
            # opposites
            # pdisp = [c2[i][0] - c2[j][0], c2[i][1] - c2[j][1]]
            if (disp[0] == -(c2[i][0] - c2[j][0]) and
                    disp[1] == -(c2[i][1] - c2[j][1])):
                col_set = adder([frozenset([i, j])], col_set)
        return col_set

    def pass_through(self, c1, c2, recursive=False):
        """returns a tuple of colliding robots, or set of tuples if
        recursive

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                if c1[i] == c2[j] and c1[j] == c2[i]:
                    col_set = adder([frozenset((i, j))], col_set)
        return col_set

    def single_bot_cross_over(self, coord1, pcoord1, coord2, pcoord2):
        """Checks for cross-over and collisions between robots one and 2
        moving from pcoord to coord

        """
        disp = (pcoord1[0] - pcoord2[0], pcoord1[1] - pcoord2[1])
        if abs(disp[1]) > 1 or abs(disp[0]) > 1:
            return False
        if (disp[0] == -(coord1[0] - coord2[0]) and
                disp[1] == -(coord1[1] - coord2[1])):
            return True
        return False

    def prio_col_check(self, coord, pcoord, t, paths=None, conn_8=False,
                       recursive=False):
        """Collision checking with paths passed as constraints

        coord  - current node
        pcoord - previous node
        t      - timestep
        paths  - paths that need to be avoided

        """
        if not isinstance(coord, tuple):
            coord = tuple(coord)
        if not isinstance(pcoord, tuple):
            pcoord = tuple(pcoord)
        if paths is not None:
            col_set = []
            adder = add_col_set
            if recursive:
                adder = add_col_set_recursive
            else:
                for i in range(len(coord)):
                    for j in range(len(paths[t])):
                        # simultaneous occupation
                        if coord[i] == paths[t][j]:
                            col_set = adder([frozenset([i])], col_set)
                            return col_set
                        # pass-through and cross-over
                        disp = [pcoord[i][0] - paths[t - 1][j][0],
                                pcoord[i][1] - paths[t - 1][j][1]]
                        if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                            continue
                        if (disp[0] == -(coord[i][0] - paths[t][j][0]) and
                                disp[1] == -(coord[i][0] - paths[t][j][1])):
                            col_set = adder([frozenset([i])], col_set)
                            return col_set
        return None


class NoRotationChecker(interface.Planner_Edge_Checker):
    """Used to wrap edge checking so more complex graphs can be cleanly
    handled (may require keeping track of state for non-trivial graphs

    Collision checking that doesn't allow rotations (i.e. robots moving
    into the place that was just vacated

    """
    def __init__(self):
        """Takes no arguments, because on grid graph, only the
        coordinates matter

        """
        pass

    def col_check(self, c1, recursive):
        """Checks for collisions at a single point.  Returns either a M*
        or rM* collision set in the form of sets, depending on the
        setting of recursive.

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                if c1[i] == c1[j]:
                    col_set = adder([frozenset([i, j])], col_set)
        return col_set

    def cross_over(self, c1, c2, recursive=False):
        """Detects cross over collisions as well as pass through
        collisions

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                # compute current displacement vector
                if c1[i] is None or c1[j] is None or c2[i] is None or c2[j] \
                        is None:
                    continue
                disp = (c1[i][0] - c1[j][0], c1[i][1] - c1[j][1])
                if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                    continue
                # Compute previous displacement vector.  Have a cross over or
                # pass through collision if the two displacement vectors are
                # opposites
                # pdisp = [c2[i][0] - c2[j][0], c2[i][1] - c2[j][1]]
                if (disp[0] == -(c2[i][0] - c2[j][0]) and
                        disp[1] == -(c2[i][1] - c2[j][1])):
                    col_set = adder([frozenset([i, j])], col_set)
                elif c1[i] == c2[j] or c1[j] == c2[i]:
                    # There is a rotation, which is banned
                    col_set = adder([frozenset([i, j])], col_set)
        return col_set


class Lazy_Edge_Checker(interface.Planner_Edge_Checker):
    """Used to wrap edge checking so more complex graphs can be cleanly
    handled (may require keeping track of state for non-trivial graphs

    """
    def __init__(self):
        """Takes no arguments, because on grid graph, only the
        coordinates matter

        """
        pass

    def col_check(self, c1, recursive):
        """Checks for collisions at a single point.  Returns either a M*
        or rM* collision set in the form of sets, depending on the
        setting of recursive.

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                if c1[i] == c1[j]:
                    col_set = adder([frozenset([i, j])], col_set)
                    return col_set
        return col_set

    def pass_through(self, c1, c2, recursive=False):
        """returns a tuple of colliding robots, or set of tuples if
        recursive

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                if c1[i] == c2[j] and c1[j] == c2[i]:
                    col_set = adder([frozenset([i, j])], col_set)
                    return col_set
        return col_set

    def cross_over(self, c1, c2, recursive=False):
        """Detects cross over collisions as well as pass through
        collisions

        """
        col_set = []
        # Select the function to be used for adding collision sets
        adder = add_col_set
        if recursive:
            adder = add_col_set_recursive
        for i in range(len(c1) - 1):
            for j in range(i + 1, len(c1)):
                # compute current displacement vector
                disp = [c1[i][0] - c1[j][0], c1[i][1] - c1[j][1]]
                if abs(disp[1]) > 1 or abs(disp[0]) > 1:
                    continue
                # Compute previous displacement vector.  Have a cross
                # over or pass through collision if the two displacement
                # vectors are opposites
                # pdisp = [c2[i][0] - c2[j][0], c2[i][1] - c2[j][1]]
                if (disp[0] == -(c2[i][0] - c2[j][0]) and
                        disp[1] == -(c2[i][1] - c2[j][1])):
                    col_set = adder([frozenset([i, j])], col_set)
                    return col_set
        return col_set
