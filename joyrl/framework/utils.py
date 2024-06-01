#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-05-26 23:54:07
LastEditor: JiangJi
LastEditTime: 2024-05-31 22:51:56
Discription: 
'''
import ray
from threading import Lock
from queue import Queue, Empty
from time import time
@ray.remote
class SharedData:
    ''' 
    '''
    def __init__(self, initial_value: float | int | dict) -> None:
        self.value = initial_value
        self.lock = Lock()

    def get_value(self):
        with self.lock:
            return self.value

    def set_value(self, new_value):
        with self.lock:
            self.value = new_value
class DeQueue(Queue):
    ''' Creating a thread-safe dequeue
    '''
    def pop(self, block=True, timeout=None):
        '''Remove and return an item from the queue.
        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until an item is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately
        available, else raise the Empty exception ('timeout' is ignored
        in that case).
        '''
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._pop()
            self.not_full.notify()
            return item
    
    def pop_nowait(self):
        '''Remove and return an item from the queue without blocking.
        Only get an item if one is immediately available. Otherwise
        raise the Empty exception.
        '''
        return self.pop(block=False)
    
    # Get an item from the queue right
    def _pop(self):
        return self.queue.pop()
    
    def append(self, item):
        ''' Put an item into the queue.
        '''
        with self.mutex:
            self.queue.append(item)
            self.unfinished_tasks += 1