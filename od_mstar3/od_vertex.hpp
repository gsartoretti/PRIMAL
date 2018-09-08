#ifndef MSTAR_OD_VERTEX_H
#define MSTAR_OD_VERTEX_H

#include <limits>

#include "mstar_type_defs.hpp"

namespace mstar{

  struct OdVertex{
    OdCoord coord;
    ColSet col_set, gen_set; // Collision set and generating collision set
    int updated; // last planning iteration used
    bool closed, open;
    double cost, h;
    OdVertex* back_ptr; // optimal way to reach this
    std::set<OdVertex*> back_prop_set; // all explored ways to reach this
    OdVertex* forwards_ptr; // way to goal from this

    OdVertex(OdCoord coord):
      coord(coord), col_set(), updated(0), closed(false), open(false),
      cost(std::numeric_limits<double>::max()), h(),
      back_ptr(nullptr), back_prop_set(), forwards_ptr(nullptr)
      {};

    bool operator>=(const OdVertex &other) const{
      return cost + h >= other.cost + other.h;
    }

    bool operator>(const OdVertex &other) const{
      return cost + h > other.cost + other.h;
    }

    bool operator<=(const OdVertex &other) const{
      return cost + h <= other.cost + other.h;
    }

    bool operator<(const OdVertex &other) const{
      return cost + h < other.cost + other.h;
    }

    /**
     * Resets a vertex used in a previous planning iteration
     *
     * @param t Current planning iteration
     */
    void reset(int t){
      if (t > updated){
	updated = t;
	open = false;
	closed = false;
	cost = std::numeric_limits<double>::max();
	back_ptr = nullptr;
	back_prop_set = std::set<OdVertex *>();
      }
    }
  };

}

#endif
