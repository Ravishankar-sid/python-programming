number_of_swaps = 0
bubble_sorted = False

while not bubble_sorted:
    bubble_sorted = True
    i = 0
    for i in range(0, len(a)):
        if i < len(a) - 1:
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]
                bubble_sorted = False
                number_of_swaps += 1

print('Array is sorted in {} swaps.'.format(number_of_swaps))
print('First Element: {}'.format(a[0]))
print('Last Element: {}'.format(a[-1]))
