"""Encapsulates the basic collision set addition functions, so they can
be accessible to any code that uses it

Also provides exceptions for indicating no solution or out of time
"""


def add_col_set_recursive(c1, c2):
    """Returns a new collision set resulting from adding c1 to c2.  No
    side effecting

    collision set is done for the recursive case, where
    ({1, 2}, ) + ({3, 4}, ) = ({1, 2}, {3, 4})

    c1, c2 - tuples of (immutable) sets

    returns:
    recursive collision set containing c1 and c2

    """
    # Make shallow copies
    c1 = list(c1)
    c2 = list(c2)
    while len(c1) > 0:
        i = 0
        # Whether c1[-1] overlaps with any element of c2
        found_overlap = False
        while i < len(c2):
            if not c2[i].isdisjoint(c1[-1]):
                # Found overlap
                if c2[i].issuperset(c1[-1]):
                    # No change in c2
                    c1.pop()
                    found_overlap = True
                    break
                # Have found a non-trivial overlap.  Need to add the
                # union to  c1 so that we can check if the union has any
                # further overlap with elements of c2
                temp = c2.pop(i)
                # replace c2[i] with the union of c2[i] and c1[-1]
                c1.append(temp.union(c1.pop()))
                found_overlap = True
                break
            else:
                # No overlap between c1[-1] and c2[i], so check next
                # element of c2
                i += 1
        if not found_overlap:
            # c1[-1] has no overlap with any element of c2, so it can be
            # added as is to c2
            c2.append(c1.pop())
    return tuple(c2)


def add_col_set(c1, c2):
    """Adds the collision sets c1 to c2.  c2 is assumed to contain a
    single,
    possibly empty, set

    c1, c2 - input collision sets

    returns:
    combined collision set containing c1 and c2

    """
    temp = frozenset([])
    if len(c2) >= 1:
        temp = c2[0]
        assert len(c2) == 1
    for i in c1:
        temp = temp.union(i)
    if len(temp) == 0:
        return ()
    return (temp, )


def col_set_add(c1, c2, recursive):
    """Adds two collision sets

    c1, c2     - input collision sets
    recursive - boolean, whether to perform recursive M* style addition

    returns:
    collision set containing c1 and c2

    """
    if recursive:
        return add_col_set_recursive(c1, c2)
    else:
        return add_col_set(c1, c2)


def effective_col_set(col_set, prev_col_set):
    """Computes the effective collision set to use given the current
    collision set and the collision set used to get to the current node

    Only makes sense when used with recursive M*

    The purpose of this code is that in recursive M*, you invoke a
    subplanner to figure out how to get to the goal, which caches the
    entire path to the goal .  The next step, you have an empty
    collision set, so you don't query the subplanner with the cached
    path, and have to find a bunch of collisions before using the cached
    solution.  This is intended for use with a memory of what the
    collision set was when you reached a given node.

    Computes the "effecitve collision set".  Elements of the memorized
    collision set are used if they have no non-empty intersections with
    elements of the current collision set that are not subsets of the
    memorized component.

    elements of col_set are NOT used if they are contained within some
    element of prev_col_set that is used.  Elements of prev_col_set are
    used if they completely contain all elements of col_set with which
    they intersect

    col_set      - current collision set
    prev_col_set - "memorized" collision set, i.e. the collision set of
                   the optimal predecessor at the time the path from the
                   optimal predecessor was first found

    returns:
    effective collision set.  Consists of the elements of the previous
    collision set, which should index subplanners which have cached
    paths available, and elements of the current collision set which
    are not contained within prev_col_set
    """
    effective_set = []
    prev_col_set = list(prev_col_set)
    col_set = list(col_set)
    while(len(prev_col_set) > 0):
        # Need to keep around the elements of col_set that won't be
        # used, because the containing element of prev_col_set may be
        # invalidated by a later element of col_set
        col_set_to_remove = []
        j = 0
        while (j < len(col_set)):
            if col_set[j].issubset(prev_col_set[-1]):
                # this element is contained in prev_col_set, so can be
                # skipped unless prev_col_set-1] is invalidated by some
                # later element of col_set
                col_set_to_remove.append(col_set.pop(j))
            elif not col_set[j].isdisjoint(prev_col_set[-1]):
                # this element partially overlaps prev_col_set,
                # invalidating it, so cannot use this element of
                # prev_col_set
                prev_col_set.pop()
                # return the elements of col_set we were going to remove
                col_set.extend(col_set_to_remove)
                break
            else:
                j += 1
        else:
            # Never broke, so prev_col_set can be used as part of the
            # effective collision set
            effective_set.append(prev_col_set.pop())
    # Just copy over any elements of col_set that survived
    effective_set.extend(col_set)
    return tuple(effective_set)


class OutOfTimeError(Exception):
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        return repr(self.value)


class NoSolutionError(Exception):
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        return repr(self.value)


class OutOfScopeError(NoSolutionError):
    def __init__(self, value=None, col_set=()):
        self.value = value
        self.col_set = col_set

    def __str__(self):
        return repr(self.value)
