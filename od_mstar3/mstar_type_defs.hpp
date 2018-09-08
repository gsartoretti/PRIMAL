#ifndef MSTAR_TYPE_DEFS
#define MSTAR_TYPE_DEFS

/**************************************************************************
 * Provides type defs that are used in multiple files
 *************************************************************************/

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <chrono>

namespace mstar{
  /**
   * Defines the graph type for individual robots.
   *
   * Assumes robot positions are indicated by integers, costs by doubles,
   * and assumes that the edge_weight property is filled
   */
  typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::bidirectionalS,  boost::no_property,
    boost::property<boost::edge_weight_t, double>> Graph;

  // type that defines the position of the robot
  typedef int RobCoord;

  // represents the coordinate of an OD node, also used to index graphs
  struct OdCoord{
    std::vector<RobCoord> coord, move_tuple;

    OdCoord(std::vector<RobCoord> in_coord, std::vector<RobCoord> in_move){
      coord = in_coord;
      move_tuple = in_move;
    }

    OdCoord(): coord(), move_tuple(){}

    bool operator==(const OdCoord &other) const{
      return (coord == other.coord) && (move_tuple == other.move_tuple);
    }

    bool is_standard() const{
      return move_tuple.size() == 0;
    }
  };

  // Holds a path in the joint configuration space
  typedef std::vector<OdCoord> OdPath;

  // defines a single set of mutually colliding robots.
  // Must be sorted in order of increasing value for logic to hold
  typedef std::set<uint> ColSetElement;

  // Defines a full collision set
  typedef std::vector<ColSetElement> ColSet;

  // defines times for checking purposes
  typedef std::chrono::system_clock Clock;
  typedef Clock::time_point time_point;
}

#endif
