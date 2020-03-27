# -*- coding: utf-8 -*-

"""[summary]
"""


class Node:
    """A Linked List is made up of Nodes, this class would be the foundation
    for our Linked List.
    """

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Main Linked List Object
    """

    def __init__(self):
        self.head = None

    def insert_head(self, data) -> None:
        """Adds value to the top (head) of the Linked List

            Arguments:
                data (any): Value to be attached to the head
        """
        # Initialize New Node
        new_node = Node(data)
        # Link New Node to 1st Element
        if self.head:
            new_node.next = self.head
        # Make new_node the 1st Element
        self.head = new_node

    def insert_tail(self, data) -> None:
        """Adds value to the bottom (tail) of the Linked List

            Arguments:
                data (any): Value to be attached to the tail
        """
        # Scan the linked list, if this is the 1st Node, initialize New Node
        if self.head is None:
            self.insert_head(data)
        else:
            temp = self.head
            # Traverse to final Node on the Linked List
            while temp.next:
                temp = temp.next
            # Create node and attach to the end of Linked List
            temp.next = Node(data)

    def print_linked_list(self) -> None:
        """Prints the current state of the Linked List
        """
        temp = self.head
        # Traverse the list to print each Node
        while temp:
            print(temp.data)
            temp = temp.next

    def delete_head(self):
        """Delete the top (head) element
        """
        temp = self.head
        if self.head:
            self.head = self.head.next
            temp.next = None

    def delete_tail(self):
        """Delete the bottom (tail) element
        """
        temp = self.head

        if self.head:
            if self.head.next is None:
                self.head = None
            else:
                while temp.next.next:
                    temp = temp.next
                temp.next, temp = None, temp.next

    def reverse_linked_list(self):
        """ Reverse the Linked List, that is, make the first node the last node
        and vice versa.
        """
        previous = None
        current = self.head

        while current:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node

        self.head = previous

    def is_list_empty(self):
        """Check wether the linked list is empty.

        Returns:
            (bool) -- True if list is empty, False if not.
        """
        return self.head is None


def main():
    A = LinkedList()

    A.insert_head(input("Add a value to Head: ").strip())
    A.insert_head(input("Add another value to Head: ").strip())

    print("Current List:")
    A.print_linked_list()

    A.insert_tail(input("Add a value to Tail: ").strip())
    A.insert_tail(input("Add another value to Tail: ").strip())

    print("Current List:")
    A.print_linked_list()

    print("Deleting Head...")
    A.delete_head()

    print("Deleting Tail...")
    A.delete_tail()

    print("Current List:")
    A.print_linked_list()

    print("Reversing Linked List...")
    A.reverse_linked_list()

    print("Current List:")
    A.print_linked_list()

    print(f"Is the current list empty? {A.is_list_empty()}")


if __name__ == "__main__":
    main()
