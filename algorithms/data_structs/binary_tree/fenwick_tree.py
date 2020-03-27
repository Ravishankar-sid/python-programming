# -*- coding: utf-8 -*-

"""Data Structures: Binary Indexed Tree AKA Fenwick Tree

This is a data structure which allows efficient element updates and calculation
of prefix sums
"""


class BinaryIndexedTree:
    """[summary]
    """

    def __init__(self, SIZE):
        self.size = SIZE
        self.ft = [0 for i in range(0, SIZE)]

    def update(self, i, value):
        """[summary]
        """
        while i < self.size:
            self.ft[i] += value
            i += i & (-i)

    def query(self, i):
        """[summary]
        """

        ret = 0

        while i > 0:
            ret += self.ft[i]
            i -= i & (-i)

        return ret


def main():
    binary_index_tree = BinaryIndexedTree(100)
    binary_index_tree.update(1, 20)
    binary_index_tree.update(4, 4)
    print(binary_index_tree.query(1))
    print(binary_index_tree.query(3))
    print(binary_index_tree.query(4))
    binary_index_tree.update(2, -5)
    print(binary_index_tree.query(1))
    print(binary_index_tree.query(3))


if __name__ == "__main__":
    main()
