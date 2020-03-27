# -*- coding: utf-8 -*-

"""Data Structures: Basic Binary Tree
"""


class Node:
    """
    A Binary Tree is made up of Nodes, this will be the foundation block for the
    binary tree
    """

    def __init__(self, data):
        # A node consists of data, left_pointer, right_pointer
        self.data = data
        self.left = None
        self.right = None


def display_tree_node(binary_tree):

    if binary_tree is None:
        return

    print(binary_tree.data)

    if binary_tree.left is not None:
        display_tree_node(binary_tree.left)

    if binary_tree.right is not None:
        display_tree_node(binary_tree.right)

    return


def binary_tree_depth(binary_tree):
    """[summary]

    Arguments:
        binary_tree {[type]} -- [description]
    """

    if binary_tree is None:
        return 0

    else:
        depth_left = binary_tree_depth(binary_tree.left)
        depth_right = binary_tree_depth(binary_tree.right)

        if depth_left > depth_right:
            return 1 + depth_left
        else:
            return 1 + depth_right


def tree_is_full_binary(binary_tree):
    """
    """
    if binary_tree is None:
        return True

    if binary_tree.left is None and binary_tree.right is None:
        return True

    if binary_tree.left is not None and binary_tree.right is not None:
        return tree_is_full_binary(binary_tree.left) and tree_is_full_binary(
            binary_tree.right
        )

    else:
        return False


def main():
    binary_tree = Node(1)
    binary_tree.left = Node(2)
    binary_tree.right = Node(3)
    binary_tree.left.left = Node(4)
    binary_tree.left.right = Node(5)
    binary_tree.right.left = Node(6)
    binary_tree.right.right = Node(7)
    binary_tree.left.left.left = Node(8)
    binary_tree.left.left.right = Node(9)
    binary_tree.right.left.left = Node(10)
    binary_tree.right.left.right = Node(11)

    print(f"Is Tree Full Binary: {tree_is_full_binary(binary_tree)}")
    print(f"The Depth of the Binary Tree is: {binary_tree_depth(binary_tree)}")
    print("Current Tree: ")
    display_tree_node(binary_tree)


if __name__ == "__main__":
    main()
