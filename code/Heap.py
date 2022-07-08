import itertools
import heapq

class Heap:
    pq: list                         # list of entries arranged in a heap
    entry_finder: dict               # mapping of tasks to entries
    REMOVED = '<removed-task>'       # placeholder for a removed task
    counter: itertools.count        # unique sequence count

    def __init__(self):
        self.pq = list()
        self.entry_finder = dict()
        self.counter = itertools.count()

    def add(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return (len(self.pq) == 0) or (self.pq[0][2] == self.REMOVED)