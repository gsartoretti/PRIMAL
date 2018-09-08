#ifndef MSTAR_COL_SET_H
#define MSTAR_COL_SET_H

#include <algorithm>

/***********************************************************************
 * Provides logic for combining collision sets
 *
 * Assumes that a collision set is of form T<T<int>> where T are
 * collections and the inner collection is sorted
 **********************************************************************/

namespace mstar{
  /**
   * tests if two sets are disjoint
   * 
   * Currently doesnt try to leverage sorted.  Empty sets will always be
   * treated as disjoint
   *
   * @param s1, s2 The sets to check
   *
   * @return True if disjoint, else false
   */
  template <class T> bool is_disjoint(const T &s1, const T &s2){
    for (auto i = s1.cbegin(); i != s1.cend(); ++i){
      for (auto j = s2.cbegin(); j != s2.cend(); ++j){
	if (*i == *j){
	  return false;
	}
      }
    }
    return true;
  };

  /**
   * Tests if s1 is a superset of s2
   *
   * Uses == to compare elements.  Does not leverage sorted values
   *
   * @param s1 potential superset
   * @param s2 potential subset
   *
   * @return True if s1 is a superset of s2, otherwise false
   */
  template <class T> bool is_superset(const T &s1, const T &s2){
    for (auto j = s2.cbegin(); j != s2.cend(); ++j){
      bool included = false;
      for (auto i = s1.cbegin(); i != s1.cend(); ++i){
	if (*i == *j){
	  included = true;
	  break;
	}
      }
      if (!included){
	return false;
      }
    }
    return true;
  };

  /**
   * specialization of is_superset that exploits sorted values
   */
  template <class T, class... extra>
  bool is_superset(const std::set<T, extra...> &s1,
		   const std::set<T, extra...> &s2){
    return std::includes(s1.cbegin(), s1.cend(), s2.cbegin(), s2.cend());
  }

  /**
   * Merges two sorted sets
   *
   * Elements of the set must be sorted.  Container of the sets must be
   * resizeable for output
   *
   */
  template <class T> T merge(const T &s1, const T &s2){
    T out(s1.size() + s2.size());
    auto it = std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
			     out.begin());
    out.resize(it - out.begin());
    return out;
  }

  template <class T, class... extra>
  std::set<T, extra...> merge(std::set<T, extra...> s1,
			      const std::set<T, extra...> &s2){
    s1.insert(s2.cbegin(), s2.cend());
    return s1;
  }

  /**
   * Adds c1 to c2
   *
   * Mutates c2
   *
   * @param c1 collision set 1
   * @param c2 collision set 2
   *
   * @return true if c2 is changed, else false
   */
  template <class T, template<class, class...> class TT, class... args>
  bool add_col_set_in_place(TT<T, args...> c1, TT<T, args...> &c2){
    bool changed = false;
    // TODO: This could be more efficient
    while (c1.size() > 0){
      int i = 0;
      // whether c1[-1] overlaps any element of c2
      bool found_overlap = false;
      while (i < c2.size()){
  	if (!is_disjoint(c2[i], c1.back())) {
  	  // found overlap
  	  if (is_superset(c2[i], c1.back())){
  	      // current element in c1 contained by the element in c2, so
  	      // the c1 element can be dropped
  	      c1.pop_back();
  	      found_overlap = true;
  	      break;
  	    }
  	  // Non-trivial overlap.  Need to add the union of the current
  	  // elements back to c1 to check if there is any further overlap
  	  // with elements of c2
	  
	  // Could just merge in place, but doubt it really matters
	  c1.back().insert(c2[i].cbegin(), c2[i].cend());
  	  c2.erase(c2.begin() + i);
	  found_overlap = true;
	  changed = true;
	  break;
	} else{
	  // no overlap between c1[-1] and c2[i], so check next element
	  // of c2
	  ++i;
	}
      }
      if (!found_overlap){
	// no overlap between c1[-1] and all elements of c2, so can
	// be added to c2 (although this will force checks against
	c2.push_back(c1.back());
	c1.pop_back();
	changed = true;
      }
    }
    return changed;
  }

  /**
   * Adds two collision sets, c1, c2
   *
   * The template monstrosity is necessary because std::vectors require two
   * parameters of which we care about one (the type), and the other is the
   * allocator.  Other containers may require more
   *
   * @param c1 collision set 1
   * @param c2 collision set 2
   *
   * @return A new collision set formed by adding c1 and c2
   */
  template <class T, template<class, class...> class TT, class... args>
  TT<T, args...> add_col_set(TT<T, args...> c1, TT<T, args...> c2){
    add_col_set_in_place(c1, c2);
    return c2;
  }

  /**
   * Computes the collision set used for expansion
   *
   * Based the generating collision set of a vertex, which is the collision
   * set of the vertex's predecessor when the predecessor was expanded.  It
   * is useful as it specifies which partial solutions have been cached.
   * For example, if the generating collision set is {{1, 2}}, then a
   * subplanner already knows how to get robots 1 and 2 to the goal, and it
   * is more efficient to directly query that subplanner, rather than set the
   * collision set to be empty.
   *
   * However, you have to account for new collisions, as stored in the
   * vertex's collision set.  If a collision set element is a subset of an
   * element of the generating collision set, use the element form the
   * generating collision set.  If a generating collision set element has
   * a non-empty intersection with a element of the collision set that is
   * not a subset, don't use that generating collision set element
   *
   * @param col_set the collision set of the vertex
   * @param gen_set the generating collision set of the vertex
   *
   * @return A new collision set to use when expanding the vertex
   */
  template <class T, template<class, class...> class TT, class... args>
  TT<T, args...> col_set_to_expand(TT<T, args...> col_set,
				   TT<T, args...> gen_set){
    TT<T, args...> ret;
    while(gen_set.size() > 0){
      // Check the last element of the generating collision set.  Either it
      // can be used, or there is a non-superset intersection, and it must
      // be removed

      // Need to keep any elements of the collision set that are subsets
      // of the generating collision set element, as a later element of the
      // collision set may invalidate the generating collision set element
      TT<T, args...> elements_to_remove;

      uint i = 0;

      bool gen_set_elem_valid = true;
      while (i < col_set.size()){
	if (is_superset(gen_set.back(), col_set[i])){
	  elements_to_remove.push_back(col_set[i]);
	  col_set.erase(col_set.begin() + i);
	} else if (!is_disjoint(gen_set.back(), col_set[i])){
	  // generating collision set element has a non-empty intersection
	  // with a collision set element that is not a sub-set, so is
	  // invalid
	  gen_set.pop_back();
	  // Need to return any collision set elements that were removed as
	  // being subsets of gen_set.back
	  col_set.insert(col_set.end(), elements_to_remove.begin(),
			 elements_to_remove.end());
	  gen_set_elem_valid = false;
	  break;
	} else{
	  i += 1;
	}
      }
      if (gen_set_elem_valid){
	ret.push_back(gen_set.back());
	gen_set.pop_back();
      }
    }
    // Any remaining collision set elements were not contained by any element
    // of the generating collision set, so should be used directly
    ret.insert(ret.end(), col_set.begin(), col_set.end());
    return ret;
  };
   
}

#endif
