#ifndef MSTAR_GRID_PLANNING_H
#define MSTAR_GRID_PLANNING_H

#include <vector>
#include <utility>

/*********************************************************************
 * Provides convienence functions for planning on 4-connected graphs
 ********************************************************************/

namespace mstar{
  /**
   * Helper function for finding paths in 4 connected paths
   *
   * The world is specified as a matrix where true indicates the presence
   * of obstacles and false indicates a clear space.  Coordinates for
   * individual robots are indicated as (row, column)
   *
   * @param obstacles matrix indicating obstacle positions.  True is obstacle
   * @param init_pos list of (row, column) pairs definining the initial
   *                 position of the robots
   * @param goals list of (row, column) pairs defining the goal configuration
   *              of the robots
   * @param inflation inflation factor used to weight the heuristic
   * @param time_limit seconds until the code declares failure
   *
   * @return Path in the joint configuration space.  Each configuration is
   *         a vector of (row, col) pairs specifying the position of
   *         individual robots
   */
  std::vector<std::vector<std::pair<int, int> > > find_grid_path(
    const std::vector<std::vector<bool> > &obstacles,
    const std::vector<std::pair<int, int> > &init_pos,
    const std::vector<std::pair<int, int> > &goals,
    double inflation, int time_limit);
}

#endif







