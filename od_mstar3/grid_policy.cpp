#include "grid_policy.hpp"

using namespace mstar;

Graph get_graph(const std::vector<std::vector<bool>> &world_map,
		const std::pair<int, int> &goal){
  int rows = (int) world_map.size();
  int columns = (int) world_map[0].size();
  typedef std::pair<int,int> E;
  std::vector<E> edges;
  std::vector<double> weights;

  std::vector<std::pair<int, int>> offsets = {{-1, 0}, {0, 1}, {1, 0},
					      {0, -1}, {0, 0}};
  for (int row = 0; row < rows; ++row){
    for (int col = 0; col < columns; ++col){
      if (world_map[row][col]){
	continue;
      }
      for (auto &off: offsets){
	int r = row + off.first;
	int c = col + off.second;
	if( r >= 0 && r < rows && c >= 0 && c < columns && ! world_map[r][c]){
	  // edge from (row, col) to (r, c)
	  // should be a more direct way, but boost is hating me
	  edges.push_back({row * columns + col, r * columns + c});
	  if (row == r && col == c && r == goal.first && c == goal.second){
	    weights.push_back(0.);
	  }else{
	    weights.push_back(1.);
	  }
	}
      }
    }
  }
  return Graph(edges.begin(), edges.end(), weights.begin(), rows * columns);
}

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
Policy mstar::grid_policy(const std::vector<std::vector<bool>> &world_map,
		   const std::pair<int, int> &goal){
  int columns = (int) world_map[0].size();
  return Policy(get_graph(world_map, goal), goal.first * columns + goal.second);
}

Policy* mstar::grid_policy_ptr(const std::vector<std::vector<bool>> &world_map,
			const std::pair<int, int> &goal){
  int columns = (int) world_map[0].size();
  return new Policy(get_graph(world_map, goal),
		    goal.first * columns + goal.second);
}
