#include <chrono>
#include <cassert>

#include "od_mstar.hpp"

using namespace mstar;

OdMstar::OdMstar(std::vector<std::shared_ptr<Policy>> policies,
		 OdCoord goals, double inflation,
		 time_point end_time, std::shared_ptr<ColChecker> col_checker){
  subplanners_ = new std::unordered_map<ColSetElement,
					std::shared_ptr<OdMstar>>();
  policies_ = policies;
  // top-level planner, so construct a set of all robot ids
  for (int i = 0; i < (int) goals.coord.size(); ++i){
    ids_.push_back(i);
  }
  goals_ = goals;
  end_time_ = end_time;
  inflation_ = inflation;
  planning_iter_ = 0;
  num_bots_ = (int) ids_.size();
  col_checker_ = col_checker;
  top_level_ = true;
}

OdMstar::OdMstar(const ColSetElement &robots, OdMstar &parent){
  subplanners_ = parent.subplanners_;
  policies_ = parent.policies_;
  for (int i: robots){
    ids_.push_back(parent.ids_[i]);
    goals_.coord.push_back(parent.goals_.coord[i]);
  }
  end_time_ = parent.end_time_;
  inflation_ = parent.inflation_;
  planning_iter_ = 0;
  num_bots_ = (int) ids_.size();
  col_checker_ = parent.col_checker_;
  top_level_ = false;
}

OdMstar::~OdMstar(){
  if (top_level_){
    delete subplanners_;
  }
}

OdPath OdMstar::find_path(OdCoord init_pos){
  reset();

  // Configure the initial vertex
  // identified by setting the back_ptr to itself
  OdVertex *first = get_vertex(init_pos);
  first->reset(planning_iter_);
  first->back_ptr = first;
  first->cost = 0;
  first->open = true;

  OpenList open_list;
  open_list.push(first);

  while (open_list.size() > 0){
    if (std::chrono::system_clock::now() > end_time_){
      throw OutOfTimeError();
    }

    OdVertex *vert = open_list.top();
    open_list.pop();
    vert->open = false;
    if (vert->closed){
      continue;
    }

    // check if this is the goal vertex
    if (vert->coord == goals_){
      vert->forwards_ptr = vert;
    }
    if (vert->forwards_ptr != nullptr){
      // Either the goal or on a previous found path to the goal
      return trace_path(vert);
    }

    expand(vert, open_list);
  }
  throw NoSolutionError();
}

void OdMstar::reset(){
  planning_iter_++;
}

double OdMstar::heuristic(const OdCoord &coord){
  // Heuristic is computed from the assigned move for elements of the
  // move tuple, and from the base coordinate for all others
  double h = 0;
  uint i = 0;
  while (i < coord.move_tuple.size()){
    h += policies_[ids_[i]]->get_cost(coord.move_tuple[i]);
    ++i;
  }
  while (i < coord.coord.size()){
    h += policies_[ids_[i]]->get_cost(coord.coord[i]);
    ++i;
  }
  return h * inflation_;
}

OdVertex* OdMstar::get_vertex(const OdCoord &coord){
  // returns a pair with the first element an interator to a <key, vertex>
  // pair and the second to a bool which is true if there was not a
  // preexisting value
  auto p = graph_.emplace(coord, coord);
  p.first->second.reset(planning_iter_);
  if (p.second){
    // new vertex, so need to set heuristic
    p.first->second.h = heuristic(coord);
  }
  return &p.first->second;
}

OdCoord get_vertex_step(OdVertex * vert){
  assert(vert != nullptr);
  while (1){
    if (vert->forwards_ptr->coord.is_standard()){
      return vert->forwards_ptr->coord;
    }
    vert = vert->forwards_ptr;
    assert(vert != nullptr);
  }
}

OdCoord OdMstar::get_step(const OdCoord &init_pos){
  OdVertex* vert = OdMstar::get_vertex(init_pos);
  if (vert->forwards_ptr != nullptr){
    return get_vertex_step(vert);
  }
  find_path(init_pos);
  return get_vertex_step(vert);
}

void OdMstar::expand(OdVertex *vertex, OpenList &open_list){
  vertex->closed = true;
  ColSet gen_set = col_set_to_expand(vertex->col_set, vertex->gen_set);
  if (gen_set.size() == 1 && (int) gen_set[0].size() == num_bots_){
    // the generating collision set contains all robots, so no caching
    // would be possible.  Therefore, don't use
    gen_set = vertex->col_set;
  }

  std::vector<OdCoord> neighbors = get_neighbors(vertex->coord, gen_set);

  // accumulates the collision sets that occur while trying to move to
  // any of the neighbors
  ColSet col_set;
  for (OdCoord &new_coord: neighbors){
    ColSet new_col = col_checker_->check_edge(vertex->coord, new_coord, ids_);
    if (!new_col.empty()){
      // State not accessible due to collisions
      add_col_set_in_place(new_col, col_set);
      continue;
    }
    
    OdVertex *new_vert = get_vertex(new_coord);
    new_vert->back_prop_set.insert(vertex);
    // Always need to at the collision set of any vertex we can reach
    // to its successors, as otherwise we would need to wait for another
    // robot to collide downstream before triggering back propagation
    add_col_set_in_place(new_vert->col_set, col_set);

    if (new_vert->closed){
      continue;
    }

    double new_cost = vertex->cost + edge_cost(vertex->coord, new_coord);
    if (new_cost >= new_vert->cost){
      continue;
    }
    new_vert->cost = new_cost;
    new_vert->back_ptr = vertex;
    new_vert->open = true;
    new_vert->gen_set = gen_set;
    open_list.push(new_vert);

    // Add an intermediate vertex's parent's col_set to its col_set, so
    // moves for later robots can be explored.  Not necessary, but should
    // reduce thrashing
    if (!new_vert->coord.is_standard()){
      add_col_set_in_place(vertex->col_set, new_vert->col_set);
    }
  }
  back_prop_col_set(vertex, col_set, open_list);
}

std::vector<OdCoord> OdMstar::get_neighbors(const OdCoord &coord,
					    const ColSet &col_set){
  // If the collision set contains all robots, invoke the non-recursive
  // base case
  if (col_set.size() == 1 && (int) col_set[0].size() == num_bots_){
    return get_all_neighbors(coord);
  }
  
  assert(coord.is_standard());

  // Generate the step along the joint policy
  std::vector<RobCoord> policy_step;
  for (int i = 0; i < num_bots_; i++){
    policy_step.push_back(policies_[ids_[i]]->get_step(coord.coord[i]));
  }

  // Iterate over colliding sets of robots, and integrate the results
  // of the sub planning for each set
  for (const ColSetElement &elem: col_set){
    // The collision set contains the local ids (relative to the robots in
    // this subplanner) of the robots in collision
    // To properly index child subplanners, need to convert to global robot
    // ids, so that the subplanners will be properly globally accessible
    ColSetElement global_col;
    for (auto &local_id: elem){
      global_col.insert(ids_[local_id]);
    }
    // Get, and if necessary construct, the appropriate subplanner.
    // returns a pair <p, bool> where bool is true if a new subplanner
    // was generated, and p is an iterator to a pair <key, val>
    if (subplanners_->find(global_col) == subplanners_->end()){
      subplanners_->insert(
	{global_col, std::shared_ptr<OdMstar>(new OdMstar(elem, *this))});
    }
    OdMstar *planner = subplanners_->at(global_col).get();
    // create the query point
    std::vector<RobCoord> new_base;
    for (const int &i: elem){
      new_base.push_back(coord.coord[i]);
    }

    OdCoord step;
    try{
      step = planner->get_step(OdCoord(new_base, {}));
    } catch(NoSolutionError &e){
      // no solution for that subset of robots, so return no neighbors
      // only likely to be relevant on directed graphs
      return {};
    }

    int elem_dex = 0;
    // now need to copy into the relevant positions in policy_step
    for (auto i: elem){
      policy_step[i] = step.coord[elem_dex];
      ++elem_dex; // could play with post appending, but don't want to
    }
  }
  return {OdCoord({policy_step}, {})};
}

std::vector<OdCoord> OdMstar::get_all_neighbors(const OdCoord &coord){
  // get the coordinate of the robot to assign a new move
  uint move_index = coord.move_tuple.size();
  std::vector<std::vector<RobCoord>> new_moves;
  for (RobCoord &move: policies_[ids_[move_index]]->get_out_neighbors(
	 coord.coord[move_index])){
    std::vector<RobCoord> new_move(coord.move_tuple);
    new_move.push_back(move);
    new_moves.push_back(new_move);
  }
  std::vector<OdCoord> ret;
  if (move_index + 1 < coord.coord.size()){
    // generating intermediate vertices
    for (auto &move_tuple: new_moves){
      ret.push_back(OdCoord(coord.coord, move_tuple));
    }
  } else {
    // generating standard vertices
    for (auto &move_tuple: new_moves){
      ret.push_back(OdCoord(move_tuple, {}));
    }
  }
  return ret;
}

double OdMstar::edge_cost(const OdCoord &source, const OdCoord &target){
  if (source.is_standard() && target.is_standard()){
    // transition between standard vertex, so all robots are assigned moves and
    // incur costs
    double cost = 0;
    for (int i = 0; i < num_bots_; ++i){
      cost += policies_[ids_[i]]->get_edge_cost(source.coord[i],
						target.coord[i]);
    }
    return cost;
  } else {
    // transition from intermediate vertex, so only one robot is assigned
    // a move and incurs cost
    uint move_index = source.move_tuple.size();
    if (target.is_standard()){
      return policies_[ids_[move_index]]->get_edge_cost(
	source.coord[move_index], target.coord[move_index]);
    } else{
      return policies_[ids_[move_index]]->get_edge_cost(
	source.coord[move_index], target.move_tuple[move_index]);
    }
  }
}

OdPath OdMstar::trace_path(OdVertex *vert){
  OdPath path;
  back_trace_path(vert, vert->forwards_ptr, path);
  forwards_trace_path(vert, path);
  return path;
}

void OdMstar::back_trace_path(OdVertex *vert, OdVertex *successor,
			      OdPath &path){
  vert->forwards_ptr = successor;
  // check if this is the final, terminal state, which is not required
  // to have a zero-cost self loop, so could get problems
  if (vert != successor){
    vert->h = successor->h + edge_cost(vert->coord, successor->coord);
  } else{
    vert->h = 0;
  }
  if (vert->coord.is_standard()){
    path.insert(path.begin(), vert->coord);
  }
  if (vert->back_ptr != vert){
    back_trace_path(vert->back_ptr, vert, path);
  }
}

void OdMstar::forwards_trace_path(OdVertex *vert, OdPath &path){
  if (vert->forwards_ptr != vert){
    if (vert->forwards_ptr->coord.is_standard()){
      path.push_back(vert->forwards_ptr->coord);
    }
    forwards_trace_path(vert->forwards_ptr, path);
  }
}

void OdMstar::back_prop_col_set(OdVertex *vert, const ColSet &col_set,
				OpenList &open_list){
  bool further = add_col_set_in_place(col_set, vert->col_set);
  if (further){
    vert->closed = false;
    if (! vert->open){
      vert->open = true;
      open_list.push(vert);
    }

    for(OdVertex *predecessor: vert->back_prop_set){
      back_prop_col_set(predecessor, vert->col_set, open_list);
    }
  }
}
