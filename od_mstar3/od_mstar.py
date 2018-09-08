"""Implementation of subdimensional expansion using operator
decomposition instead of vanilla A*, with better graph abstraction.
All coordinates are to be tuples and all collision sets are to be lists
of immutable sets (frozenset). This partial rewrite will focus on
converting everything that can be immutable into an immutable structure

Intended to support both mstar and rMstar."""


import od_mstar3.workspace_graph as workspace_graph
import sys
import time as timer  # So that we can use the time command in ipython
import od_mstar3.SortedCollection as SortedCollection
from od_mstar3.col_set_addition import add_col_set_recursive, add_col_set
from od_mstar3.col_set_addition import effective_col_set
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError, col_set_add
#from itertools import izip # No need since Python3, use global function zip()
#try:
#    import ipdb as pdb
#except ImportError:
#    # Default to pdb
#    import pdb


MAX_COST = workspace_graph.MAX_COST
PER_ROBOT_COST = 1  # Cost a robot accrues for not being at its goal position
POSITION = 0
MOVE_TUPLE = 1  # Tuple of destination coordinate tuples for each robot's move

global_move_list = []  # Used for visualization

def find_path(obs_map, init_pos, goals, recursive=True, inflation=1.0,
              time_limit=5 * 60.0, astar=False, get_obj=False, connect_8=False,
              full_space=False, return_memory=False, flood_fill_policy=False,
              col_checker=None, epemstar=False, makespan=False,
              col_set_memory=True):
    """Finds a path in the specified obstacle environment from the
    initial position to the goal.

    obs_map           - obstacle map,  matrix with 0 for free,  1 for
                        obstacle
    init_pos          - ((x1, y1), (x2, y2), ...) coordinates of the
                        initial state, should be tuples
    goals             - ((x1, y1), (x2, y2), ...) coordinates of the goal
                        should be tuples
    recursive         - True for rM*,  false for basic M*
    inflation         - factor by which the metric will be inflated
    time_limit        - how long to run before raising an error
                        (declaring timeout)
    astar             - use basic A* instead of operator decomposition to
                        search the graph produced by M* (i.e. run M* not
                        ODM*)
    get_obj           - Return the Od_Mstar instance used in path
                        planning, default False
    connect_8         - True (default) for 8 connected graph,  False for
                        4 connected graph
    full_space        - If True,  run pure A* or OD (depending on the
                        astar flag) instead of subdimensional expansion.
                        Default False
    return_memory     - Returns information on memory useage.
                        Default False
    flood_fill_policy - compute policy with flood fill instead of
                        resumable A*
    col_checker       - Optional custom collision checker object,  used
                        for searching non-grid graphs.  Default None
    epemstar          - Use EPEA* to search the graph rather than A* or
                        OD
    makespan          - minimize makespan (time to solution),
                        instead of minimizing time robots spend away
                        from their robots
    col_set_memory    - remember previous step collision set, intended
                        to provide more efficient cached path
                        utillization.  True by default
    """
    global global_move_list
    if (col_checker is None or isinstance(col_checker,
                                          workspace_graph.Edge_Checker)):
        goals = tuple(map(tuple, goals))
        init_pos = tuple(map(tuple, init_pos))
    global_move_list = []
    o = Od_Mstar(obs_map, goals, recursive=recursive, inflation=inflation,
                 astar=astar, connect_8=connect_8, full_space=full_space,
                 flood_fill_policy=flood_fill_policy, col_checker=col_checker,
                 epeastar=epemstar, makespan=makespan,
                 col_set_memory=col_set_memory)
    # Need to make sure that the recursion limit is great enough to
    # actually construct the path
    longest = max([o.sub_search[(i, )].get_cost(init_pos[i])
                   for i in range(len(init_pos))])
    # Guess that the longest path will not be any longer than 5 times the
    # longest individual robot path
    sys.setrecursionlimit(max(sys.getrecursionlimit(), longest * 5 *
                              len(init_pos)))
    path = o.find_path(init_pos, time_limit=time_limit)
    num_nodes = o.get_memory_useage(False)
    corrected_mem = o.get_memory_useage(True)
    if get_obj:
        return path, o
    if return_memory:
        return path, num_nodes, corrected_mem
    return path


class Od_Mstar(object):
    """Implements M* and rM* using operator decomposition instead of
    basic M* as the base computation.

    """
    def __init__(self, obs_map, goals, recursive, sub_search=None,
                 col_checker=None, rob_id=None, inflation=1.0,
                 end_time=10 ** 15, connect_8=False, astar=False,
                 full_space=False, flood_fill_policy=False, epeastar=False,
                 offset_increment=1, makespan=False, col_set_memory=False):
        """
        obs_map           - obstacle map,  matrix with 0 for free,  1
                            for obstacle
        goals             - ((x1, y1), (x2, y2), ...) coordinates of the
                            goal, should be tuples
        recursive         - True for rM*, false for basic M*
        sub_search        - Sub planners, should be None for the full
                            configuration space
        col_checker       - object to handle robot-robot collision
                            checking.  Should implement the same
                            interface as workspace_graph.Edge_Checker
        rob_id            - maps local robot identity to full
                            configuration space identity,  should be
                            None for the full configuration space
                            instance
        inflation         - how much the metric should be inflated by
        end_time          - when the search should be terminated
        connect_8         - True for 8 connected graph,  False for 4
                            connected graph
        astar             - use basic A* instead of operator
                            decomposition
        full_space        - whether to perform a full configuration
                            space search
        flood_fill_policy - compute policy with flood fill instead of
                            resumable A*
        epeastar          - Uses EPEA* instead of OD or A* for graph
                            search
        offset_increment  - how much to increase the EPEA* offset after
                            every expansion
        makespan          - minimize makespan (time to solution),
                            instead of minimizing time robots spend away
                            from their robots
        col_set_memory    - remember previous step collision set,
                            intended to provide more efficient cached
                            path utillization.  False by default
        """
        # visualize - turn on visualization code - DISABLED
        self.obs_map = obs_map
        self.recursive = recursive
        self.sub_search = sub_search
        # Stores the global ids of the robots in order of their position
        # in coord
        self.rob_id = rob_id
        self.goals = goals
        # Graph that holds the graph representing the joint configuration space
        self.graph = {}
        self.end_time = end_time
        self.inflation = float(inflation)
        self.connect_8 = connect_8
        self.astar = astar
        self.epeastar = epeastar
        self.offset_increment = offset_increment
        self._makespan = makespan

        # Store some useful values
        self.updated = 0
        self.num_bots = len(goals)
        # self.visualize = visualize
        self.full_space = full_space
        # Need a different key incorporating the offset for EPEM*
        if self.epeastar:
            self.open_list_key = lambda x: (-x.cost - x.h * self.inflation -
                                            x.offset)
        else:
            self.open_list_key = lambda x: -x.cost - x.h * self.inflation
        if self.rob_id is None:
            self.rob_id = tuple(range(len(goals)))
        self.col_checker = col_checker
        if self.col_checker is None:
            self.col_checker = workspace_graph.Edge_Checker()
        self.flood_fill_policy = flood_fill_policy
        # Making everything that can be immutable,  immutable
        self._col_set_memory = col_set_memory
        self.gen_policy_planners(sub_search, self.obs_map, self.goals)

    def gen_policy_planners(self, sub_search, obs_map, goals):
        """Creates the sub-planners and necessary policy keys.  This is
        because pretty much every sub-planner I've made requires
        adjusting the graph used to create the policies and passing
        around dummy sub_searches

        side effects to generate self.sub_search and self.policy_keys
        """
        self.policy_keys = tuple([(i, ) for i in self.rob_id])
        self.sub_search = sub_search
        if self.sub_search is None:
            self.sub_search = {}
            # Wrapping the robot number in a tuple so we can use the same
            # structure for planners for compound robots
            if self.flood_fill_policy:
                for dex, key in enumerate(self.policy_keys):
                    self.sub_search[key] = workspace_graph.Workspace_Graph(
                        obs_map, goals[dex], connect_8=self.connect_8)
            else:
                for dex, key in enumerate(self.policy_keys):
                    self.sub_search[key] = workspace_graph.Astar_Graph(
                        obs_map, goals[dex], connect_8=self.connect_8,
                        makespan=self._makespan)

    def get_graph_size(self, correct_for_size=True):
        """Returns the number of nodes in the current graph"""
        if correct_for_size:
            return len(self.graph) * len(self.rob_id)
        return len(self.graph)

    def get_memory_useage(self, correct_for_size=True):
        """Returns the total number of nodes allocated in this planner
        and any subplanners.
        """
        temp_sum = self.get_graph_size(correct_for_size)
        for i in self.sub_search:
            temp_sum += self.sub_search[i].get_graph_size()
        return temp_sum

    def reset(self):
        """resets the map for later searches,  does not remove
        forwards_pointer
        """
        self.updated += 1

    def heuristic(self, coord, standard_node):
        """Returns the heuristic value of the specified coordinate.

        Does not handle inflation naturally so we can update the
        heuristic properly

        coord         - coordinate of the node at which to compute the
                        heuristic
        standard_node - whether this is a standard node
        """
        if standard_node:
            cost = sum(self.sub_search[key].get_cost(coord[dex])
                       for dex, key in enumerate(self.policy_keys))
            # return self.inflation * cost
            return cost
        else:
            # Compute heuristic for robots which have moved
            cost = sum(self.sub_search[key].get_cost(coord[MOVE_TUPLE][dex])
                       for dex, key in enumerate(
                           self.policy_keys[:len(coord[MOVE_TUPLE])]))
            # compute heuristic for robots which have not moved
            cost += sum(self.sub_search[key].get_cost(
                coord[POSITION][dex + len(coord[MOVE_TUPLE])])
                for dex, key in enumerate(self.policy_keys[len(coord[
                    MOVE_TUPLE]):]))
            return cost

    def pass_through(self, coord1, coord2):
        """Tests for a collision during transition from coord 1 to coord
        2.

        coord1, coord2 - joint coordinates of multirobot system

        returns:

        collision set for the edge,  empty list if there are no
        collisions
        """
        # return self.col_checker.pass_through(coord1, coord2, self.recursive)
        return self.col_checker.cross_over(coord1, coord2, self.recursive)

    def incremental_col_check(self, start_coord, new_coord):
        """Performs an incremental collision check for new coord.

        Assumes that the position of a single new robot has been added to
        a list of coordinates of robots.  Checks whether adding this new
        robot will lead to a collision.  Start coord is the joint state
        before the action being built in new_coord,  and may contain more
        robots than new_coord. counts on the implementation of the
        incremental collision checks to be intelligent to avoid issues

        start_coord - coordinate at which the system starts
        new_coord   - coordinate to which the system moves

        returns:

        collision_set formed form the colliding robots during the move
        """
        col_set = self.col_checker.incremental_cross_over(
            start_coord, new_coord, self.recursive)
        if col_set:
            return col_set
        return self.col_checker.incremental_col_check(
            new_coord, self.recursive)

    def get_node(self, coord, standard_node):
        """Returns the node at the specified coordinates.

        Remember intermediate nodes are of the form
        (base_coord, move_tuple)

        coord         - coordinates of the node,  potentially an
                        intermediate node
        standard_node - whether this is a standard node or an
                        intermediate node
        """
        if coord in self.graph:
            # Node already exists.  reset if necessary
            t_node = self.graph[coord]
            t_node.reset(self.updated)
            return t_node
        # Need to instantiate the node
        if standard_node:
            col = self.col_checker.col_check(coord, self.recursive)
        else:
            # Only check for collisions between robots whose move has
            # been determined
            col = self.col_checker.col_check(coord[MOVE_TUPLE], self.recursive)
        free = (len(col) == 0)
        t_node = mstar_node(coord, free, self.recursive, standard_node)
        # Cache the resultant col_set
        t_node.col_set = col
        t_node.updated = self.updated
        t_node.h = self.heuristic(coord, standard_node)
        # Add the node to the graph
        self.graph[coord] = t_node
        return t_node

    def get_step(self, init_pos, standard_node=True):
        """Get the optimal step from init_pos.

        Computes the entire optimal path if necessary, but preferentially
        relying on the cached paths stored in mstar_node.forwards_ptr.

        init_pos      - coordinate of the node to compute the step from
        standard_node - standard_node whether init_pos represents a
                        standard node

        returns:

        coordinate of the optimal step towards the goal
        """
        cur_node = self.get_node(init_pos, standard_node)
        temp = cur_node.get_step()
        if temp is not None:
            return temp
        # Use a zero time limit,  so the end time will not be modified
        path = self.find_path(init_pos, time_limit=-1)
        return cur_node.get_step()

    def gen_init_nodes(self, init_pos):
        """Generate the initial search nodes.

        Potentially more than one node is generated, but in practice
        will usually just one will be generated

        init_pos - initial position

        returns:

        list of initial nodes
        """
        first = self.get_node(init_pos, True)
        first.open = True
        first.cost = 0
        first.back_ptr = first
        return [first]

    def find_path(self, init_pos, time_limit=5 * 60):
        """Finds a path from init_pos to the goal specified when self
        was instantiated.

        init_pos   - ((x1, y1), (x2, y2), ...) coordinates of initial
                     position
        time_limit - time allocated to find a solution.  Will raise an
                     exception if a path cannot be found within this time
                     period
        """
        self.reset()
        if time_limit > 0:
            self.end_time = timer.time() + time_limit
            # For replanning to work correctly, need to update the end
            # time for all subplanners.  Otherwise, the end time of the
            # subplanners will never be updated, so if you make a query
            # more than the original time_limit seconds after the first
            # query to this object, you will always get a timeout,
            # regardless of the time limit used on the second query
            for planner in self.sub_search.values():
                if hasattr(planner, 'end_time'):
                    planner.end_time = self.end_time

        # Configure the goal node
        goal_node = self.get_node(self.goals, True)
        goal_node.forwards_ptr = goal_node
        # Use the negation of the cost,  so SortedCollection will put the
        # lowest value item at the right of its internal list
        init_nodes = self.gen_init_nodes(init_pos)
        open_list = SortedCollection.SortedCollection(init_nodes,
                                                      key=self.open_list_key)

        while len(open_list) > 0:
            if timer.time() > self.end_time:
                raise OutOfTimeError(timer.time())
            node, consistent = open_list.consistent_pop()
            if not consistent:
                continue
            node.open = False
            if self.solution_condition(node):
                path = node.get_path()
                return tuple(path)
            self.expand(node, open_list)
        raise NoSolutionError()

    def solution_condition(self, node):
        """Checks whether we have finished finding a path when node has
        been reached

        Checks whether node.forwards_ptr indicates that a path to the
        goal has been found

        node - node to check for indicating a path to the goal

        returns:

        True if goal has been reached or a cached path to the goal has
        been reached, else False
        """
        if node.forwards_ptr is not None:
            return True

        return False

    def expand(self, node, open_list):
        """Handles the expansion of the given node and the addition of
        its neighbors to the open list

        node      - node to expand
        open_list - open list used during the search
        """
        node.closed = True
        # ASSUMES THAT get_neighbors HANDLES UPDATING NEIGHBOR COST,
        # AND DOES NOT RETURN NEIGHBORS FOR WHICH THERE IS ALREADY A
        # PATH AT LEAST AS GOOD
        if self.recursive:
            neighbors,  col_set = self.get_neighbors_recursive(node)
        else:
            neighbors,  col_set = self.get_neighbors_nonrecursive(node)

        # node is the only element in the backpropagation sets of
        # neighbors that has changed,  so we can backpropagate from here
        old_col_set = node.col_set
        if not self.full_space:
            node.back_prop_col_set(col_set, open_list, epeastar=self.epeastar)
        for i in neighbors:
            i.back_ptr = node
            # Even if the node is already in the open list,  removing if
            # from its old position (given by the old cost value) is too
            # expensive, requiring an O(N) operation to delete.  Simply
            # add the new value and reject the old copy (which will be
            # marked as closed),  when you come to it
            i.open = True
            open_list.insert_right(i)
        if self.epeastar:
            # if running epeastar
            if old_col_set == node.col_set:
                # If the collision set changed,  then adding the node
                # back to the open list with properly updated collision
                # set has been handled by the backprop function
                node.offset += self.offset_increment
                open_list.insert(node)

    def od_mstar_neighbors(self, node):
        """Generates the free neighbors of the given node for the
        non-recursive case, using operator decomposition

        Also returns the associated collision set due to neighbors
        which are non-free due to robot-robot collisions.  Only returns
        nodes which can be most cheaply reached through node

        node - node to determine neighbors

        returns:

        (neighbors, col_set)
        neighbors - collision free neighbors which can most efficiently
                    be reached from node
        col_set   - collision set for neighbors which are not collision
                    free
        """
        col_set = ()
        if not node.free:
            # Can't have an out neighbor for a node in collision
            return col_set, node.col_set
        rob_dex = 0  # Keeps track of the robot to move in this step

        # split the coordinates into the start coordinate and the move
        # list if the node is standard,  doing this so variables are
        # initialized in  the preferred namespace,  which is probably not
        # necessary
        move_list = ()
        start_coord = node.coord
        if not node.standard_node:
            start_coord = node.coord[POSITION]
            move_list = node.coord[MOVE_TUPLE]
            rob_dex = len(node.coord[MOVE_TUPLE])
        if ((len(node.col_set) > 0 and rob_dex in node.col_set[0]) or
                self.full_space):
            # This robot is in the collision set,  so consider all
            # possible neighbors
            neighbors = self.sub_search[
                self.policy_keys[rob_dex]].get_neighbors(start_coord[rob_dex])
        else:
            neighbors = [self.sub_search[self.policy_keys[rob_dex]].get_step(
                start_coord[rob_dex])]
        # check if this is the last robot to be moved
        filled = (rob_dex == (self.num_bots - 1))

        new_neighbors = []
        # visualize_holder = []
        for i in neighbors:
            # Generate the move list with the new robot position
            new_moves = list(move_list)
            new_moves.append(i)
            new_moves = tuple(new_moves)
            # Check for collisions in the transition to the new
            # position, only need to consider the robots in the move list
            # pass through
            pass_col = self.pass_through(start_coord[:rob_dex + 1], new_moves)
            if len(pass_col) > 0:
                # Have robot-robot collisions
                col_set = col_set_add(pass_col, col_set, self.recursive)
                continue
            # Need to branch on whether we have filled the move list
            if filled:
                # Generate a standard node.  Static collisions are found
                # in self.get_node()
                new_node = self.get_node(new_moves, True)
            else:
                # Generate an intermediate node
                new_node = self.get_node((start_coord, new_moves), False)
            if node not in new_node.back_prop_set:
                new_node.back_prop_set.append(node)
            # Always need to add the col_set of any vertex that we can
            # actually reach,  as otherwise,  we would need to wait for
            # another robot to collide downstream of the reached vertex
            # before that vertex would back propagate its col_set
            col_set = col_set_add(new_node.col_set, col_set, self.recursive)
            if not new_node.free:
                continue
            # Skip if closed
            if new_node.closed:
                continue
            # Handle costs, which depends soely on the move list,
            # function to allow for alternate cost functions
            temp_cost = self.od_mstar_transition_cost(start_coord, node.cost,
                                                      i, rob_dex)
            if temp_cost >= new_node.cost:
                continue
            new_node.cost = temp_cost
            new_neighbors.append(new_node)
            # Set the intermediate nod's col_set equal to its parent,
            # so later elements will actually be explored.  Not
            # technically required but will cut back on thrashing
            if not new_node.standard_node:
                new_node.add_col_set(node.col_set)
        return new_neighbors, col_set

    def od_mstar_transition_cost(self, start_coord, prev_cost, neighbor,
                                 rob_dex):
        """Computes the transition cost for a single robot in od_mstar
        neighbor generation

        start_coord - base position of robots (prior to move assignment)
        prev_cost   - cost of base node
        neighbor    - proposed move assignmetn
        rob_dex     - robot move is assigned to

        returns:

        cost of a single robot transitioning state
        """
        prev_cost += self.sub_search[self.policy_keys[rob_dex]].get_edge_cost(
            start_coord[rob_dex], neighbor)
        return prev_cost

    def gen_epeastar_coords(self, node):
        """Helper function for generating neighbors of a node using EPEA*

        Uses a two step process. First the incremental costs are
        computed, then the neighbors fitting those incremental costs.
        More directly matches what was done in the EPEA* paper.  Performs
        incremental collision checking during the generation of
        neighbors,  to prune out as many invalid nodes as early as
        possible

        node - node for which to generate neighbors
        """
        adder = add_col_set
        if self.recursive:
            adder = add_col_set_recursive
        offset = node.offset
        coord = node.coord
        if len(node.col_set) == 0:
            # have empty collision set
            new_coord = tuple(
                self.sub_search[self.policy_keys[dex]].get_step(
                    coord[dex]) for dex in range(self.num_bots))
            pass_col = self.pass_through(coord, new_coord)
            if pass_col:
                return [], pass_col
            col = self.col_checker.col_check(new_coord, self.recursive)
            if col:
                return [], col
            return [new_coord], []
        search_list = [(0, ())]
        assert len(node.col_set) == 1
        node_col = node.col_set[0]
        for rob_dex in range(self.num_bots):
            if rob_dex in node_col:
                offsets = self.sub_search[
                    self.policy_keys[rob_dex]].get_offsets(coord[rob_dex])
            else:
                offsets = (0, )
            new_list = []
            for cost, pos in search_list:
                for off in offsets:
                    if rob_dex < self.num_bots - 1:
                        if off + cost <= offset:
                            new_list.append((off + cost, pos + (off, )))
                    elif off + cost == offset:
                        # For the last robot,  only want to keep costs which
                        # match perfectly
                        new_list.append((off + cost, pos + (off, )))
                search_list = new_list
        neighbors = []
        col_set = []
        for offset, costs in search_list:
            gen_list = [()]
            for dex, c in enumerate(costs):
                if dex in node_col:
                    neib = (self.sub_search[
                            self.policy_keys[dex]].get_offset_neighbors(
                            coord[dex], c))
                else:
                    neib = ((0, self.sub_search[
                        self.policy_keys[dex]].get_step(coord[dex])),)
                new_list = []
                for _, n in neib:
                    for old in gen_list:
                        new_coord = old + (n, )
                        # Perform collision checking
                        tcol = self.incremental_col_check(coord, new_coord)
                        if tcol:
                            col_set = adder(col_set, tcol)
                            continue
                        new_list.append(new_coord)
                gen_list = new_list
            neighbors.extend(gen_list)
        return neighbors, col_set

    def get_epeastar_neighbors(self, node):
        """Generates the free neighbors of the given node for the
        non-recursive case.

        Also returns the associated collision set due to neighbors
        which are non-free due to robot-robot collisions.  Only returns
        nodes which can be most cheaply reached through node

        node - node to be expanded

        returns:
        (neighbors, col_set)
        neighbors - neighbors that can most be efficiently reached from
                    node, that are collision free
        col_set   - collisions incurred when trying to reach
                    non-collision free nodes
        """
        if not node.free:
            # Can't have an out neighbor for a node in collision
            return [], node.col_set
        start_coord = node.coord
        neighbor_coords, col_set = self.gen_epeastar_coords(node)
        neighbors = []
        for i in neighbor_coords:
            new_node = self.get_node(i, True)
            if node not in new_node.back_prop_set:
                new_node.back_prop_set.append(node)
            if not new_node.free:
                continue
            # update costs
            if new_node.closed:
                continue
            t_cost = self.epeastar_transition_cost(start_coord, node.cost, i)
            if t_cost < new_node.cost:
                new_node.cost = t_cost
                new_node.back_ptr = node
                neighbors.append(new_node)
        return neighbors, col_set

    def epeastar_transition_cost(self, start_coord, prev_cost, new_coord):
        """Computes the cost of a new node at the specified coordinates,
        starting from the given position and cost

        start_coord - node at which the system starts
        prev_cost   - cost of the node at start_coord
        new_coord   - destination node
        """
        for dex, (source, target) in enumerate(zip(start_coord, new_coord)):
            prev_cost += self.sub_search[self.policy_keys[dex]].get_edge_cost(
                source, target)
        return prev_cost

    def get_neighbors_nonrecursive(self, node):
        """Generates neighbors using a non-recursive method.  Note that
        collision sets will still be generated in the style specified by
        self.recursive

        node - node for which to generate neighbors
        """
        if self.astar:
            return self.get_astar_neighbors(node)
        elif self.epeastar:
            return self.get_epeastar_neighbors(node)
        return self.od_mstar_neighbors(node)

    def create_sub_search(self, new_goals, rob_id):
        """Creates a new instance of a subsearch for recursive search

        new_goals - goals for the subset of the robots
        rob_ids   - ids of the robots involved in the subsearch

        returns:

        new OD_Mstar instance to perform search for the specified subset
        of robots"""
        return Od_Mstar(self.obs_map, new_goals, self.recursive,
                        sub_search=self.sub_search,
                        col_checker=self.col_checker, rob_id=rob_id,
                        inflation=self.inflation,
                        end_time=self.end_time, connect_8=self.connect_8,
                        astar=self.astar, full_space=self.full_space,
                        epeastar=self.epeastar, makespan=self._makespan,
                        col_set_memory=self._col_set_memory)

    def get_subplanner_keys(self, col_set):
        """Returns keys to subplanners required for planning for some
        subset of robots.

        col_set - collision set to be solved

        returns:

        keys for the necessary subplanners in self.sub_search
        """
        # Convert the collision sets into the global indicies,  and
        # convert to tuples.  Assumes self.rob_id is sorted
        global_col = list(map(lambda y: tuple(map(lambda x: self.rob_id[x], y)),
                         col_set))
        # generate the sub planners,  if necessary
        for dex, gc in enumerate(global_col):
            if gc not in self.sub_search:
                t_goals = tuple([self.goals[k] for k in col_set[dex]])
                self.sub_search[gc] = self.create_sub_search(t_goals, gc)
        return global_col

    def get_neighbors_recursive(self, node):
        """Get the neighbors of node for recursive M*.

        Uses operator decomposition style expansion when necessary,  may
        fail when called on an intermediate node

        node - node for which to generate neighbors


        returns:
        (neighbors, col_set)
        neighbors - list of coordinates for neighboring, reachable
                    nodes
        col_set   - collisions generated by trying to transition to
                    non-reachable neighbors
        """
        # Handle collision set memory if necessary
        # use_memory = False
        if self._col_set_memory:
            col_set = effective_col_set(node.col_set, node.prev_col_set)
            effective_set = col_set
            # if set(col_set) != set(node.col_set):
            #     # using memory
            #     use_memory = True
            # Sort the collision set,  which also converts them into
            # lists
            col_set = list(map(sorted, col_set))
        else:
            # Sort the collision set,  which also converts them into lists
            col_set = list(map(sorted, node.col_set))
        # Use standard operator decomposition,  if appropriate
        if len(col_set) == 1 and len(col_set[0]) == self.num_bots:
            # At base of recursion case
            return self.get_neighbors_nonrecursive(node)
        start_coord = node.coord
        if not node.standard_node:
            assert False
        # Generate subplanners for new coupled groups of robots and get
        # their sub_search keys
        coupled_keys = self.get_subplanner_keys(col_set)
        # Generate the individually optimal step
        new_coord = [self.sub_search[self.policy_keys[i]].get_step(
            start_coord[i]) for i in range(self.num_bots)]
        # Iterate over the colliding sets of robots,  and integrate the
        # results of the sup planning for each set
        for i in range(len(col_set)):
            # if use_memory and frozenset(col_set[i]) in node.prev_col_set:
                # assert self.sub_search[
                #     coupled_keys[i]].graph[
                #         tuple([start_coord[j]
                #                for j in col_set[i]])].forwards_ptr != None
            try:
                new_step = self.sub_search[coupled_keys[i]].get_step(
                    tuple([start_coord[j] for j in col_set[i]]))
            except NoSolutionError:
                # Can't get to the goal from here
                return [], []
            # Copy the step into position
            for j in range(len(col_set[i])):
                new_coord[col_set[i][j]] = new_step[j]

        new_coord = tuple(new_coord)
        # process the neighbor
        pass_col = self.pass_through(start_coord, new_coord)
        if len(pass_col) > 0:
            # Have collisions before reaching node
            return [], pass_col
        new_node = self.get_node(new_coord, True)
        if node not in new_node.back_prop_set:
            new_node.back_prop_set.append(node)
        if not new_node.free:
            return [],  new_node.col_set
        # Skip if closed
        if new_node.closed:
            return [],  new_node.col_set
        # Compute the costs. THIS MAY NOT WORK IF node IS AN INTERMEDIATE
        # NODE
        t_cost = self.get_node(start_coord, True).cost
        t_cost = self.od_rmstar_transition_cost(start_coord, t_cost,
                                                new_node.coord)
        if t_cost < new_node.cost:
            new_node.cost = t_cost
            if self._col_set_memory:
                new_node.prev_col_set = effective_set
            return [new_node], new_node.col_set
        return [], new_node.col_set

    def od_rmstar_transition_cost(self, start_coord, prev_cost, new_coord):
        """Computes the transition cost for a single robot in od_rmstar
        neighbor generation

        start_coord - base position of robots (prior to move assignment)
        prev_cost   - cost of base node
        new_coord    - proposed move assignmetn

        returns:

        total cost of reaching new_coord via start_coord
        """
        for dex, (source, target) in enumerate(zip(start_coord, new_coord)):
            prev_cost += self.sub_search[self.policy_keys[dex]].get_edge_cost(
                source, target)
        return prev_cost

    def alt_get_astar_neighbors(self, node):
        """Gets neighbors of a specified node using the standard A*
        approach.


        assumes working with standard nodes

        node - node for which to generate neighbors

        returns:
        (neighbors, col_set)
        neighbors - coordinates of collision free neighboring nodes
        col_set   - collisions resulting from trying to reach
                    non-collision free neighbors
        """
        start_coord = node.coord
        # Generate the individually optimal setp
        base_coord = [self.sub_search[self.policy_keys[i]].get_step(
            start_coord[i]) for i in range(self.num_bots)]
        old_coords = [base_coord]
        assert len(node.col_set) <= 1
        to_explore = node.col_set
        if self.full_space:
            to_explore = [range(self.num_bots)]
        for i in to_explore:
            for bot in i:
                new_coords = []
                neighbors = self.sub_search[self.policy_keys[bot]]\
                                .get_neighbors(start_coord[bot])
                for neigh in neighbors:
                    for k in old_coords:
                        temp = k[:]
                        temp[bot] = neigh
                        new_coords.append(temp)
                old_coords = new_coords
        col_set = []
        neighbors = []
        old_coords = list(map(tuple, old_coords))
        for i in old_coords:
            # Check if we can get there
            pass_col = self.pass_through(start_coord, i)
            if len(pass_col) > 0:
                col_set = col_set_add(pass_col, col_set, self.recursive)
                continue
            new_node = self.get_node(i, True)
            col_set = col_set_add(new_node.col_set, col_set, self.recursive)
            if node not in new_node.back_prop_set:
                new_node.back_prop_set.append(node)
            if not new_node.free:
                continue
            # update costs
            if new_node.closed:
                continue
            t_cost = node.cost
            for j in range(len(start_coord)):
                t_cost += self.sub_search[self.policy_keys[j]].get_edge_cost(
                    start_coord[i], new_node.coord[j])
            if t_cost < new_node.cost:
                new_node.cost = t_cost
                new_node.back_ptr = node
                neighbors.append(new_node)
        return neighbors, col_set

    def get_astar_neighbors(self, node):
        """Gets neighbors of a specified node using the standard A*
        approach,

        assumes working with standard nodes

        node - node for which to generate neighbors

        returns:
        (neighbors, col_set)
        neighbors - coordinates of collision free neighboring nodes
        col_set   - collisions resulting from trying to reach
                    non-collision free neighbors
        """
        start_coord = node.coord
        # Generate the individually optimal setp
        base_coord = [self.sub_search[self.policy_keys[i]].get_step(
            start_coord[i]) for i in range(self.num_bots)]
        old_coords = [base_coord]
        assert len(node.col_set) <= 1
        to_explore = node.col_set
        if self.full_space:
            to_explore = [range(self.num_bots)]
        for i in to_explore:
            for bot in i:
                new_coords = []
                neighbors = self.sub_search[self.policy_keys[bot]]\
                                .get_neighbors(start_coord[bot])
                for neigh in neighbors:
                    for k in old_coords:
                        temp = k[:]
                        temp[bot] = neigh
                        new_coords.append(temp)
                old_coords = new_coords
        col_set = []
        neighbors = []
        old_coords = list(map(tuple, old_coords))
        for i in old_coords:
            # First check if this path is relevant.  I.e. if there is already a
            # better path to the node,  then the search will never try to use
            # that route,  so we don't need to consider collisions
            new_node = self.get_node(i, True)
            if node.free:
                t_cost = node.cost
                for j in range(len(start_coord)):
                    t_cost += self.sub_search[
                        self.policy_keys[j]].get_edge_cost(start_coord[j],
                                                           i[j])
                if t_cost >= new_node.cost:
                    continue
            # Check if we can get there
            pass_col = self.pass_through(start_coord, i)
            if len(pass_col) > 0:
                col_set = col_set_add(pass_col, col_set, self.recursive)
                continue
            new_node = self.get_node(i, True)
            col_set = col_set_add(new_node.col_set, col_set, self.recursive)
            if node not in new_node.back_prop_set:
                new_node.back_prop_set.append(node)
            if not new_node.free:
                continue
            # update costs
            if new_node.closed:
                continue
            if t_cost < new_node.cost:
                new_node.cost = t_cost
                new_node.back_ptr = node
                neighbors.append(new_node)
        return neighbors, col_set


class mstar_node(object):
    """Holds the data needed for a single node in operator decomposition
    m* coord should be a tuple of tuples.  Standard nodes have
    coordinates of the form ((x1, y1), (x2, y2), ...),  while
    intermediate nodes have coordinates of the form (((x1, y1), ...),
    move_tuple)
    """

    __slots__ = ['free', 'coord', 'updated', 'open', 'closed', 'standard_node',
                 'h', 'cost', 'back_ptr', 'back_prop_set', 'col_set',
                 'recursive', 'forwards_ptr', 'assignment', 'colset_changed',
                 'offset', 'prev_col_set']

    def __init__(self, coord, free, recursive, standard_node, back_ptr=None,
                 forwards_ptr=None):
        """Constructor for mstar_node

        Assumes the col_set is empty by default

        coord         - tuple giving coordinates,  may store partial
                        moves if not standard node
        free          - part of the free configuration space
        standard_node - represents a standard node,  and not a partial
                        move
        back_ptr      - pointer to best node to get to self
        forwards_ptr  - pointer along the best path to the goal
        """
        self.free = free
        self.coord = coord
        self.updated = -1
        # Whether already in the open list
        self.open = False

        # Whether this has been expanded.  Note that a node can be added
        # back to the open list after it has been expanded,  but will
        # still be marked as closed.  It cannot have its cost changed,
        # but it can add neighbors, but not be added as a neighbor
        self.closed = False
        self.standard_node = standard_node
        # Heuristic cost to go,  None to ensure it will be properly
        # calculated
        self.h = None
        # Cost to reach
        self.cost = MAX_COST

        # Optimal way to reach this node.  Point to self to indicate the
        # initial position
        self.back_ptr = back_ptr
        self.back_prop_set = []  # Ways found to reach this node
        self.col_set = ()
        # store the collision set of back_ptr when the path from
        # back_ptr to self was first found.  Used for hopefully more
        # efficient cached path access
        self.prev_col_set = ()
        self.recursive = recursive

        # Keeps track of solutions that have already been found,
        # replace forwards_tree.  Denote the goal node by pointing
        # forwards_ptr
        # to itself
        self.forwards_ptr = forwards_ptr
        self.assignment = None  # Used for multiassignment mstar

        # Used to track whether new assignments need to be generated for
        # MURTY  mstar
        self.colset_changed = False
        # Tracks current offset for multiple re-expansion a la EPEA*
        self.offset = 0

    def reset(self, t):
        """Resets if t > last update time"""
        if t > self.updated:
            self.updated = t
            self.open = False
            self.closed = False
            self.cost = MAX_COST
            self.back_ptr = None
            self.back_prop_set = []
            self.offset = 0

    def get_path(self):
        """Gets the path passing through path to the goal,  assumes that
        self is either the goal node,  or a node connected to the goal
        node through forwards_pointers
        """
        path = self.backtrack_path()
        return self.forwards_extend_path(path)

    def backtrack_path(self, path=None, prev=None):
        """Finds the path that leads up to this node,  updating
        forwards_ptr so that we can recover this path quickly,  only
        returns standard nodes

        path - current reconstructed path for use in recusion, must
               start as None
        prev - pointer to the last node visited by backtrack_path, used
               to update forwards_ptr to record the best paths to the
               goal
        """
        if path is None:
            path = []
        if prev is not None:
            self.forwards_ptr = prev
            if isinstance(self.h, tuple):
                # Needed for constrained od_mstar,  and don't feel like
                # coming up with a better solution for now
                self.h = (prev.h[0] + prev.cost[0] - self.cost[0], self.h[1])
            else:
                self.h = prev.h + (prev.cost - self.cost)
        if self.standard_node:
            assert self.coord not in path
            path.insert(0, self.coord)
        if self.back_ptr == self:
            # Done so that it cannot terminate on a node that wasn't
            # properly initialized
            return path
        return self.back_ptr.backtrack_path(path, self)

    def forwards_extend_path(self, path):
        """Extends the path from self to the goal node,  following
        forwards pointers,  only includes standard nodes

        path - current path to extend towards the goal, as list of joint
               configuration space coordinates
        """
        if self.forwards_ptr == self:
            return path
        if self.forwards_ptr.standard_node:
            path.append(self.forwards_ptr.coord)
        return self.forwards_ptr.forwards_extend_path(path)

    def add_col_set(self, c):
        """Adds the contents of c to self.col_set.

        c - collision set to add to the current node's collision set

        returns:

        True if modifications were made, else False
        """
        if len(c) == 0:
            return False
        if self.recursive:
            temp = add_col_set_recursive(c, self.col_set)
        else:
            temp = add_col_set(c, self.col_set)
        modified = (temp != self.col_set)
        if modified:
            self.col_set = temp
            return True
        return False

    def back_prop_col_set(self, new_col, open_list, epeastar=False):
        """Propagates the collision dependencies found by its children
        to the parent,  which adds any new dependencies to this col_set

        new_col   - the new collision set to add
        open_list - the open list to which nodes with changed collisoin
                    sets are added,  assumed to be a SortedCollection
        """
        further = self.add_col_set(new_col)
        if further:
            self.colset_changed = True
            if not self.open:
                # assert self.closed
                self.open = True
                # self.closed = False
                self.offset = 0

                # Inserting to the left of any node with the same key
                # value,  to encourage exploring closer to the collison
                open_list.insert(self)
            elif epeastar and self.offset != 0:
                # Need to reset the offset and reinsert to allow a path
                # to be found even if the node is already in the open
                # list
                self.offset = 0
                # Inserting to the left of any node with the same key
                # value, to encourage exploring closer to the collison
                open_list.insert(self)
            for j in self.back_prop_set:
                j.back_prop_col_set(self.col_set, open_list, epeastar=epeastar)

    def get_step(self):
        """Returns the coordinate of the next standard node in the path,

        returns:

        None if no such thing
        """
        if self.forwards_ptr is None:
            return None
        if self.forwards_ptr.standard_node:
            return self.forwards_ptr.coord
        else:
            return self.forwards_ptr.get_step()


def individually_optimal_paths(obs_map, init_pos, goals):
    """Returns the individually optimal paths for a system"""

    path = []
    for i in range(len(init_pos)):
        path.append(find_path(obs_map, [init_pos[i]], [goals[i]]))
    # Need to convert to full space
    max_length = max(list(map(len, path)))
    for i in path:
        while len(i) < max_length:
            i.append(i[-1])
    jpath = []
    for i in range(max_length):
        temp = []
        for j in path:
            temp.append(j[i][0])
        jpath.append(temp)
    return jpath


def find_path_limited_graph(obs_map, init_pos, goals, recursive=True,
                            inflation=1.0, time_limit=5 * 60.0, astar=False,
                            get_obj=False, connect_8=False, full_space=False,
                            return_memory=False, flood_fill_policy=False,
                            pruning_passes=5):
    global global_move_list
    global_move_list = []
    o = Od_Mstar(obs_map, goals, recursive=recursive, inflation=inflation,
                 astar=astar, connect_8=connect_8, full_space=full_space,
                 flood_fill_policy=flood_fill_policy)
    import prune_graph
    G = prune_graph.to_networkx_graph(obs_map)
    for i in range(pruning_passes):
        G = prune_graph.prune_opposing_edge(G, num_edges=5)
    # Replace the individual policies with limited graphs
    for i in range(len(o.goals)):
        o.sub_search[(i, )] = workspace_graph.Networkx_Graph(
            obs_map, goals[i], graph=G, connect_8=connect_8)
    # Need to make sure that the recursion limit is great enough to
    # actually construct the path
    longest = max([o.sub_search[(i, )].get_cost(init_pos[i])
                   for i in range(len(init_pos))])
    # Guess that the longest path will not be any longer than 5 times the
    # longest individual robot path
    sys.setrecursionlimit(max(sys.getrecursionlimit(), longest * 5 *
                              len(init_pos)))
    path = o.find_path(init_pos, time_limit=time_limit)
    num_nodes = o.get_memory_useage(False)
    corrected_mem = o.get_memory_useage(True)
    if get_obj:
        return path, o
    # if visualize:
    #     return path,  global_move_list
    if return_memory:
        return path, num_nodes, corrected_mem
    return path
