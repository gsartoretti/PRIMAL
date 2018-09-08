#include <vector>
#include <utility>
#include <memory>

#include "grid_planning.hpp"
#include "grid_policy.hpp"
#include "od_mstar.hpp"
#include "mstar_type_defs.hpp"

using namespace mstar;

/**
 * Converts from (row, column) coordinates to vertex index
 */
OdCoord to_internal(std::vector<std::pair<int, int>> coord,
				  int cols){
  std::vector<RobCoord> out;
  for (auto &c: coord){
    out.push_back(c.first * cols + c.second);
  }
  return OdCoord(out, {});
};

/**
 * Converts from vertex index to (row, column) format
 */
std::vector<std::pair<int, int>> from_internal(OdCoord coord,
					       int cols){
  std::vector<std::pair<int, int>> out;
  for (auto &c: coord.coord){
    out.push_back({c / cols, c % cols});
  }
  return out;
};

std::vector<std::vector<std::pair<int, int>>> mstar::find_grid_path(
  const std::vector<std::vector<bool>> &obstacles,
  const std::vector<std::pair<int, int>> &init_pos,
  const std::vector<std::pair<int, int>> &goals,
  double inflation, int time_limit){
  // compute time limit first, as the policies fully compute 
  // Need to convert time limit to std::chrono format
  time_point t = std::chrono::system_clock::now();
  t += Clock::duration(std::chrono::seconds(time_limit));

  int cols = (int) obstacles[0].size();
  OdCoord _init = to_internal(init_pos, cols);
  OdCoord _goal = to_internal(goals, cols);
  std::vector<std::shared_ptr<Policy>> policies = {};
  for (const auto &goal: goals){
    policies.push_back(std::shared_ptr<Policy>(
			 grid_policy_ptr(obstacles, goal)));
  }
  OdMstar planner(policies, _goal, inflation, t,
		  std::shared_ptr<ColChecker>(new SimpleGraphColCheck()));
  OdPath path = planner.find_path(_init);
  std::vector<std::vector<std::pair<int, int>>> out;
  for (auto &coord: path){
    out.push_back(from_internal(coord, cols));
  }
  return out;
}
