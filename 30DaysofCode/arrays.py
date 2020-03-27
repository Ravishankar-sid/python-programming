# -*- coding: utf-8 -*-

n = int(input())

arr = list(map(int, input().rstrip().split()))

print(" ".join(map(str, arr[::-1])))
