# -*- coding: utf-8 -*-

import math


def check_prime(x):
    if x <= 1:
        return False
    elif x == 2:
        return True
    elif x > 2 and x % 2 == 0:
        return False
    md = math.floor(math.sqrt(x))
    for i in range(3, 1 + md, 2):
        if x % i == 0:
            return False
    return True


x = int(input())

a = []

for i in range(0, x):
    a.append(int(input()))
for i in a:
    if check_prime(i):
        print("Prime")
    else:
        print("Not prime")
