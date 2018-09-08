#ifndef MSTAR_GRID_POLICY_H
#define MSTAR_GRID_POLICY_H

/**************************************************************************
 * Generates policy for grid maps
 **************************************************************************/

#include <vector>
#include <utility>

#include "mstar_type_defs.hpp"
#include "policy.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

namespace mstar{

  /**
   * Generates a policy for a 4 connected grid
   *
   * The internal coordinates are of the form row * num_rows + col
   * Allows for weighting at the goal for free
   *
   * @param world_map matrix of values describing grid true for obstacle,
   *                  false for clear
   * @param goal (row, column) of goal
   *
   * @return Policy object describing problem
   */
  Policy grid_policy(const std::vector<std::vector<bool>> &world_map,
		     const std::pair<int, int> &goal);

  Policy* grid_policy_ptr(const std::vector<std::vector<bool>> &world_map,
			  const std::pair<int, int> &goal);
}

#endif
