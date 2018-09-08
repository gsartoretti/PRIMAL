"""This module defines interfaces for the low-level graphs and
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

    3.  Configuration graph edge checking, which determines whether
        moving between two configurations is permissible.  For example,
        configuration graph edge checking should not allow a robot to
        move out of bounds of the workspace.

    4.  Planner edge checking, which determines whether moving between
        two states of robot positions will result in any collisions.
        For example, planner edge checking should check to see if two
        robots pass through each other as they move between positions.
"""


class Graph_Interface(object):

    """Interface for configuration space generators

    This graph interface enumerates the methods that any
    configuration space generator should implement.  These graphs are
    used by policy graphs such as A*.
    """

    def get_edge_cost(self, coord1, coord2):
        """Returns edge_cost of going from coord1 to coord2."""
        raise NotImplementedError

    def get_neighbors(self, coord):
        """Returns the collision free neighbors of the specified coord.

        Return value is a list of tuples each of which are a coordinate
        """
        raise NotImplementedError

    # This is a function to return the in neighbors of a coordinate.
    # Designed by default to handle un-directed graphs
    get_in_neighbors = get_neighbors


class Policy_Interface(object):

    """Interface showing required implemented functions for all policies

    This interface enumerates the functions that must be exposed by
    policies for M* to function correctly. A policy object with this
    interface provides a route for a single robot.  Underneath the policy
    interface is a graph object which describes the configuration space
    through which robots can move.  The underlying graph object does all
    of the work of calculating the configuration space based on the
    actual environment in which the robot is moving

    **All config inputs must be hashable**
    """

    def get_cost(self, config):
        """Returns the cost of moving from given position to goal"""
        raise NotImplementedError

    def get_edge_cost(self, config1, config2):
        """Returns the cost of traversing an edge in the underlying
        graph
        """
        raise NotImplementedError

    def get_step(self, config):
        """Returns the configurations of the optimal neighbor of config"""
        raise NotImplementedError

    def get_neighbors(self, config):
        """Returns neighboring configurations of config

        This function returns the configurations which are next to
        config

        Return list of tuples, each of which is a coordinate
        """
        raise NotImplementedError

    def get_graph_size(self, correct_for_size=True):
        """Returns number of nodes in graph"""
        raise NotImplementedError

    def get_limited_offset_neighbors(self, config, max_offset, min_offset=0):
        """Returns set of neighbors between the offset arguments"""
        raise NotImplementedError

    def get_offset_neighbors(self, config, offset):
        """Returns neighbors of coord with offset specified by argument"""
        raise NotImplementedError

    def get_offsets(self, config):
        """Return the offsets of the neighbors"""
        raise NotImplementedError


class Config_Edge_Checker(object):
    """Checks robot collisions with objects and edges of workspace"""

    def col_check(self, state, recursive):
        """Checks for collisions at a single state

        state     - list of coordinates of robots
        recursive - generate collisions sets for rM*

        Returns:
        M* collision set in type set if recursive false
        rM* collision set in type set if recursive true
        """
        raise NotImplementedError


class Planner_Edge_Checker(object):
    """Checks for robot collisions on an edge in a planner's graph

    Currently, no methods have to be implemented because the collision
    methods change based on the graph.
    """

    def pass_through(self, state1, state2, recursive=False):
        """Detects pass through collisions

        state1 - list of robot coordinates describing initial state
        state2 - list of robot coordinates describing final state,

        Returns:
            M* collision set in type set if recursive false
            rM* collision set in type set if recursive true
        """
        raise NotImplementedError

    def col_check(self, state, recursive):
        """Checks for collisions at a single state

        state     - list of coordinates of robots
        recursive - generate collisions sets for rM*

        Returns:
            M* collision set in type set if recursive false
            rM* collision set in type set if recursive true
        """
        raise NotImplementedError

    def cross_over(self, state1, state2, recursive=False):
        """Detects cross over and pass through collisions


        state1 - list of robot coordinates describing initial state
        state2 - list of robot coordinates describing final state

        Returns:
            M* collision set in type set if recursive false
            rM* collision set in type set if recursive true
        """
        raise NotImplementedError

    def simple_pass_through(self, state1, state2):
        """Check for pass through collisions

        state1 - list of robot coordinates describing initial state
        state2 - list of robot coordinates describing final state

        Returns:
        True if pass through collision
        False otherwise
        """
        raise NotImplementedError

    def simple_col_check(self, state):
        """Checks for robot-robot collisions at state,

        state - list of robot coordinates

        returns:
        True if collision
        False otherwise
        """
        raise NotImplementedError

    def simple_cross_over(self, state1, state2):
        """Check for cross over collisions in 8-connected worlds

        state1 - list of robot coordinates describing initial state
        state2 - list of robot coordinates describing final state

        returns:
        True if collision exists
        False otherwise
        """
        raise NotImplementedError

    def simple_incremental_cross_over(self, state1, state2):
        """Check for cross over collisions in 8-connected worlds.

        Assumes that collision checking has been performed for everything
        but the last robot in the coordinates.  To be used to save a bit
        of time for partial expansion approaches

        state1 - list of robot coordinates describing initial state
        state2 - list of robot coordinates describing final state

        returns:
        True if collision exists
        False otherwise
        """
        raise NotImplementedError

    def simple_incremental_col_check(self, state1):
        """Checks for robot-robot collisions at c1,

        Assumes that collision checking has been performed for everything
        but the last robot in the coordinates.  To be used to save a bit
        of time for partial expansion approaches

        state1 - list of robot coordinates

        returns:
        True if collision exists
        False otherwise
        """
        raise NotImplementedError

    def single_bot_outpath_check(self, cur_coord, prev_coord, cur_t, paths):
        """Tests for collisions from prev_coord to cur_coord

        Checks for cross over collisions and collisions at the same
        location when moving from cur_coord to prev_coord while robots
        are moving in paths

        cur_coord - position of a single robot

        Returns:

        True if collision exists
        False otherwise
        """
        raise NotImplementedError

    def simple_prio_col_check(self, coord, t, paths, pcoord=None,
                              conn_8=False):
        """Returns true, if collision is detected, false otherwise
        at the moment only used to check the obstacle collisions, but
        didn't want to reject the other code already

        coord - coord of potential new neighbor
        t - current time step
        paths - previously found paths
        pcoord - previous coordinate of the path

        Returns:
        True if collision exists
        False otherwise
        """
        raise NotImplementedError

    def incremental_col_check(self, state, recursive):
        """Checks for robot-robot collisions in state

        state     - list of coordinates of robots
        recursive - generate collisions sets for rM*

        Only checks whether the last robot is
        involved in a collision, for use with incremental methods

        Returns:
            M* collision set in type set if recursive false
            rM* collision set in type set if recursive true
        """
        raise NotImplementedError

    def incremental_cross_over(self, state1, state2, recursive=False):
        """Detects cross over collisions as well as pass through
        collisions.

        Only checks if the last robot is involved in a collision, for use
        with partial expansion approaches.

        state1 - list of robot coordinates describing initial state
        state2 - list of robot coordinates describing final state,

        Returns:
            M* collision set in type set if recursive false
            rM* collision set in type set if recursive true
        """
        raise NotImplementedError

    def single_bot_cross_over(self, coord1, pcoord1, coord2, pcoord2):
        """Checks for cross-over and collisions between robots 1 and 2

        Robots are moving from pcoord to coord

        pcoord1 - first position of first robot
        coord1  - second position of first robot
        pcoord2 - first position of second robot
        coord2  - second position of second robot

        Returns:
        True if collision
        False otherwise
        """
        raise NotImplementedError

    def prio_col_check(self, coord, pcoord, t, paths=None, conn_8=False,
                       recursive=False):
        """Collision checking with paths passed as constraints

        coord  - current node
        pcoord - previous node
        t      - timestep
        paths  - paths that need to be avoided

        Returns: (collision sets are of type set)
            M* collision set if collision exists and recursive is false
            rM* collision set if collision exists and recursive is true
            None if no collision exists
        """
        raise NotImplementedError
