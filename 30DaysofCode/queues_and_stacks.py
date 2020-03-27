# -*- coding: utf-8 -*-


class Solution:
    def __init__(self):
        self.stack = list()
        self.queue = list()

    def pushCharacter(self, char):
        self.stack.append(char)

    def enqueueCharacter(self, char):
        self.queue.append(char)

    def popCharacter(self):
        return self.stack.pop(-1)

    def dequeueCharacter(self):
        return self.queue.pop(0)
