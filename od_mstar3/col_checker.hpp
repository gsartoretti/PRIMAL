#ifndef MSTAR_COL_CHECKER_H
#define MSTAR_COL_CHECKER_H

#include "mstar_type_defs.hpp"

namespace mstar{

  class ColChecker{
  public:
    virtual ~ColChecker(){};
    virtual ColSet check_edge(const OdCoord &c1, const OdCoord &c2,
			      const std::vector<int> ids) const = 0;
  };

  /**
   * Collision checker for simple bidirected graphs, where no edges overlap
   *
   * I.e. for pebble motion on the graph where you only have to worry about
   * robots swapping positions, and not about diagonals crossing.  Allows
   * for rotations
   */
    class SimpleGraphColCheck: public ColChecker{
    public:
      /**
       * Checks for collision while traversing the edge from c1 to c2
       *
       * Finds collisions both while traversing the edge and when at the
       * goal configuration.
       *
       * @param c1 the source coordinate of the edge
       * @param c2 the target coordinate of the edge
       * @param ids list of global robot ids.  Necessary for heterogeneous
       *            robots
       *
       * @return the collision set containing the colliding robots
       */
      ColSet check_edge(const OdCoord &c1, const OdCoord &c2,
			const std::vector<int> ids) const;
  };
};

#endif
