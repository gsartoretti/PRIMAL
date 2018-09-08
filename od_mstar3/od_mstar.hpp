#ifndef MSTAR_OD_MSTAR_H
#define MSTAR_OD_MSTAR_H

#include <unordered_map>
#include <functional>
#include <queue>
#include <memory>
#include <exception>

#include <boost/functional/hash_fwd.hpp>

#include "mstar_type_defs.hpp"
#include "col_set.hpp"
#include "od_vertex.hpp"
#include "col_checker.hpp"
#include "policy.hpp"

namespace std{
  template <> struct hash<mstar::OdCoord>{
    size_t operator()(const mstar::OdCoord &val) const{
      size_t hash = boost::hash_range(val.coord.cbegin(), val.coord.cend());
      boost::hash_combine<size_t>(
	hash,
	boost::hash_range(val.move_tuple.cbegin(), val.move_tuple.cend()));
      return hash;
    }
  };

  template <> struct hash<std::vector<int>>{
    size_t operator()(const std::vector<int> &val) const{
      return boost::hash_range(val.cbegin(), val.cend());
    }
  };

  template <> struct hash<mstar::ColSetElement>{
    size_t operator()(const mstar::ColSetElement &val) const{
      return boost::hash_range(val.cbegin(), val.cend());
    }
  };
}


namespace mstar{

  struct greater_cost{
    bool operator()(const mstar::OdVertex *x, const mstar::OdVertex *y) const{
      if (x == nullptr || y == nullptr){
	return true;
      }
      return *x > *y;
    }
  };

  // Sort in decreasing order to give cheap access to the cheapest elements
  typedef std::priority_queue<OdVertex*, std::vector<OdVertex*>,
			      greater_cost> OpenList;

  class OdMstar {    
  public:
    /**
     * Constructs a new, top level M* planner
     *
     * @param policies pointer to vector of policies.
     *                 OdMstar does not take ownership
     * @param goals goal configuration of entire system
     * @param inflation inflation factor
     * @param end_time time at which M* will declare failure
     * @param checker collision checking object
     */
    OdMstar(
      std::vector<std::shared_ptr<Policy>> policies,
      OdCoord goals, double inflation, time_point end_time,
      std::shared_ptr<ColChecker> col_checker);

    /**
     * Creates a subplanner for a subsest of the robots
     *
     * robots is a collision set element in the frame of parent, not global
     * robot ids
     */
    OdMstar(const ColSetElement &robots, OdMstar &parent);

    ~OdMstar();

    /**
     * Computes the optimal path to the goal from init_pos
     *
     * @param init_pos coordinate of the initial joint configuration
     *
     * @return the path in the joint configuration graph to the goal
     *
     * @throws OutOfTimeError ran out of planning time
     * @throws NoSolutionError no path to goal from init_pos
     */
    OdPath find_path(OdCoord init_pos);

  private:
    /**TODO: fix
     * This is kind of horrifying, but I cannot store the OdMstar objects
     * directly in the unordered map, as I get ungodly errors that look
     * like they come from an allocator.  Adding copy constructor and
     * assignment operator doesn't work, so its something involved about
     * STL.  Think this works, but annoying
     */
    std::unordered_map<ColSetElement, std::shared_ptr<OdMstar>> *subplanners_;
    std::vector<std::shared_ptr<Policy>> policies_;
    // ids of the robots this planner handles.  Assumed to be in ascending
    // order
    std::vector<int> ids_;
    OdCoord goals_;
    // holds the nodes in the joint configuration space
    std::unordered_map<OdCoord, OdVertex> graph_;
    time_point end_time_; // When planning will be halted
    double inflation_; // inflation factor for heuristic
    int planning_iter_; // current planning iteration
    int num_bots_;
    std::shared_ptr<ColChecker> col_checker_;
    bool top_level_; // tracks if the top level planner

    OdMstar(const OdMstar &that) = delete;

    /**
     * Resets planning for a new planning iteration.
     *
     * Does not reset forwards_ptrs, as those should be valid across
     * iterations
     */
    void reset();

    /**
     * Computes the heuristic value of a vertex at a given coordinate
     *
     * @param coord coordinate for which to compute a heuristic value
     *
     * @return the (inflated) heuristic value
     */
    double heuristic(const OdCoord &coord);

    /**
     * Returns a reference to the vertex at a given coordinate
     *
     * this->graph retains ownership of the vertex.  Will create the vertex
     * if it does not already exist.
     *
     * @param coord coordinate of the desired vertex
     *
     * @return pointer to the vertex at coord.
     */
    OdVertex* get_vertex(const OdCoord &coord);

    /**
     * Returns the optimal next step from init_pos
     *
     * Will compute the full path if necessary, but preferentially uses
     * cached results in forwards_ptrs.  Expected to only be called from
     * a standard coordinate, and to only return a standard coordinate
     *
     * @param init_pos coordinate to compute the optimal next step from
     *
     * @returns the coordinate of the optimal next step towards the goal
     */
    OdCoord get_step(const OdCoord &init_pos);

    /**
     * Generates the neighbors of vertex and add them to the open list
     *
     * @param vertex OdVertex to expand
     * @param open_list the sorted open list being used
     */
    void expand(OdVertex *vertex, OpenList &open_list);

    /**
     * Returns the limited neighbors of coord using recursive calculation
     *
     * @param coord Coordinates of vertex to generate neighbor thereof
     * @param col_set collision set of vertex to generate neighbors
     *
     * @return list of limited neighbors
     */
    std::vector<OdCoord> get_neighbors(
      const OdCoord &coord, const ColSet &col_set);

    /**
     * Returns the limited neighbors of coord using non-recursive computation
     *
     * Called when the collision set contains all of the robots, as a base
     * case for get_neighbors, thus always generate all possible neighbors
     *
     * @param coord Coordinates of vertex to generate neighbor thereof
     *
     * @return list of limited neighbors
     */
     std::vector<OdCoord> get_all_neighbors(
       const OdCoord &coord);

    /**
     * Returns the cost of traversing a given edge
     *
     * @param source coordinate of the source vertex
     * @param target coordinate of the target vertex
     *
     * @return the cost of the edge
     */
    double edge_cost(const OdCoord &source, const OdCoord &target);

    /**
     * Returns the path through a vertex
     *
     * Assumes that back_ptr and forwards_ptr are set and non-none at vert
     * Identifies each end of the path by looking for a back_ptr/forwards_ptr
     * pointed at the holder
     *
     * @param vert the vertex to trace a path through
     *
     * @return the path passing through vert containing only standard vertices
     */
    OdPath trace_path(OdVertex *vert);

    /**
     * Generates the path to the specified vertex
     *
     * Sets forward_ptrs to cache the path, and updates the heuristic
     * values of the vertices on the path so we can end the moment a
     * vertex on a cached path is expanded.
     *
     * TODO: double check that making the heuristic inconsistent in this
     * fashion is OK.
     *
     * @param vert the vertex to trace the path to
     * @param successor the successor of vert on the path
     * @param path place to construct path
     */
    void back_trace_path(OdVertex *vert, OdVertex *successor, OdPath &path);

    /**
     * Genertes the path from the specified vertex to the goal
     *
     * Non-trivial only if vert lies on a previously cached path
     *
     * @param vert the vertex to trace the path from
     * @param path place to construct path
     */
    void forwards_trace_path(OdVertex *vert, OdPath &path);

    /**
     * Backpropagates collision set information to all predecessors of a
     * vertex.
     *
     * Adds vertices whose collision set changes back to the open list
     *
     * @param vertex pointer to the vertex to back propagate from
     * @param col_set the collision set that triggered backpropagation
     * @param open_list the current open list
     */
    void back_prop_col_set(OdVertex *vert, const ColSet &col_set,
			   OpenList &open_list);
  };  

  struct OutOfTimeError : public std::exception{
    const char * what () const throw(){
      return "Out of Time";
    }
  };

  struct NoSolutionError : public std::exception{
    const char * what () const throw(){
      return "No Solution";
    }
  };

};

#endif
