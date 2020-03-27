"""Python Implementation of a Selection Sort Algorithm

To run locally:
    python3 selection_sort.py
"""


def selection_sort(collection):
    """Implementation of the Selection Sort Algorithm

    Arguments:
        collection -- User Input of elements which need to be sorted.

    Returns:
        (list) -- Sorted List of input elements
    """
    length = len(collection)

    for i in range(length - 1):
        least = i
        for j in range(i + 1, length):
            if collection[j] < collection[least]:
                least = j
        if least != i:
            collection[least], collection[i] = (collection[i], collection[least])

    return collection


if __name__ == "__main__":
    input_user = input("Enter a collection of numbers, seperated by a Comma:\n").strip()

    unsorted = [int(item) for item in input_user.split(",")]
    print(selection_sort(unsorted))
