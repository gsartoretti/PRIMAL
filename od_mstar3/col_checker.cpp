#include "col_checker.hpp"
#include "col_set.hpp"

using namespace mstar;

// /**
//  * Performs simple pebble motion on the graph collision checking
//  *
//  * @param c1 source
//  * @param c2 target
//  *
//  * @return collision set of the edge
//  */
// template<class T>
// ColSet simple_edge_check(const T &c1,
// 			 const T&c2){
//   ColSet col;
//   for (uint i = 0; i < c1.size(); i++){
//     for (uint j = i; j < c1.size(); j++){
//       if (c2[i] == c2[j] || (c1[i] == c2[j] && c1[j] == c2[i])){
// 	add_col_set_in_place({{i, j}}, col);
//       }
//     }
//   }
//   return col;
// }

/**
 * Iterator version
 */
template<class T>
ColSet simple_edge_check(T source_start, T source_end,
			 T target_start, T target_end){
  int size = source_end - source_start;
  ColSet col;
  for (uint i = 0; i < size; i++){
    for (uint j = i + 1; j < size; j++){
      if (*(target_start + i) == *(target_start + j) ||
	  (*(source_start + i) == *(target_start + j) &&
	   *(source_start + j) == *(target_start + i))){
	add_col_set_in_place({{i, j}}, col);
      }
    }
  }
  return col;
}

ColSet SimpleGraphColCheck::check_edge(const OdCoord &c1,
				       const OdCoord &c2,
				       const std::vector<int> ids) const{
  if (c2.is_standard()){
    return simple_edge_check(c1.coord.cbegin(), c1.coord.cend(),
			     c2.coord.cbegin(), c2.coord.cend());
  }
  // c2 is an intermediate vertex, so only check for collisions between
  // robots with an assigned move in c2
  int size = c2.move_tuple.size();
  return simple_edge_check(c1.coord.cbegin(), c1.coord.cbegin() + size,
			   c2.move_tuple.cbegin(), c2.move_tuple.cend());
}
