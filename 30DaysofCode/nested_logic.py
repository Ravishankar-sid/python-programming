def fine_calculator(deadline_for_return, returned_on):
    #[Start fine_calculator]
    if returned_on[2] > deadline_for_return[2]:
        return 10000

    elif (returned_on[2] == deadline_for_return[2] and
        returned_on[1] > deadline_for_return[1]):

        return 500 * (returned_on[1] - deadline_for_return[1])

    elif (returned_on[2] == deadline_for_return[2] and
        returned_on[1] == deadline_for_return[1] and
        returned_on[0] > deadline_for_return[0]):

        return 15 * (returned_on[0] - deadline_for_return[0])

    else:
        return 0
    #[End fine_calculator]

# Run Program
if __name__ == '__main__':
    returned_on = list(map(int, input().split()))
    deadline_for_return = list(map(int, input().split()))

    fine = fine_calculator(deadline_for_return, returned_on)
    print(fine)
