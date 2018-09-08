from threading import Lock, Condition

class GroupLock:
    '''Queues asynchronus threads by group.

    Args:

        groups (key list list) : a list of lists of keys which represent the keys for each thread in each group.

                                e.g. -> [['thread1','thread2'], ['thread3']]'''
    def __init__(self, groups):
        self.groups = groups

        self.activeGroup = 0

        self.hasReleased = [{member:False for member in group} for group in groups]
        self.numGroups = len(groups)

        self._groupConditions = [{member:Condition(Lock()) for member in group} for group in groups]

    def acquire(self, group, id):
        '''Acquires the lock, blocks if not the thread's turn yet. A thread can acquire the lock once per group cycle.

        e.g.
        if GroupLock was initialized with [['thread1','thread2'], ['thread3']], a calls would be

            GroupLock.acquire(1, 'thread3')
            GroupLock.acquire(0, 'thread2')


        Args:

            group (int): number of calling thread's group
            id    (key): key given to identify the thread (given in init)'''
        self._groupConditions[group][id].acquire()
        if self.hasReleased[group][id] or self.activeGroup != group:
            self._groupConditions[group][id].wait()

    def release(self, group, id):
        '''Releases the group lock. All threads in a lock must release before the next group's turn can begin.

        e.g.
        if GroupLock was initialized with [['thread1','thread2'], ['thread3']], a calls would be

            GroupLock.acquire(1, 'thread3')
            GroupLock.acquire(0, 'thread2')


        Args:

            group (int): number of calling thread's group
            id    (key): key given to identify the thread(given in init)'''
        self._groupConditions[group][id].release()
        self.hasReleased[group][id] = True

        if all(self.hasReleased[group].values()):
            self.hasReleased[group] = {member:False for member in self.hasReleased[group]}
            self.activeGroup = (self.activeGroup + 1) % len(self.groups)

            releasedGroup = self.activeGroup
            for memberCondition in self._groupConditions[releasedGroup]:
                self._groupConditions[releasedGroup][memberCondition].acquire()
                self._groupConditions[releasedGroup][memberCondition].notify_all()
                self._groupConditions[releasedGroup][memberCondition].release()

    def releaseAll(self):
        releasedGroup = self.activeGroup
        for memberCondition in self._groupConditions[releasedGroup]:
            self._groupConditions[releasedGroup][memberCondition].acquire()
            self._groupConditions[releasedGroup][memberCondition].notify_all()
            self._groupConditions[releasedGroup][memberCondition].release()
