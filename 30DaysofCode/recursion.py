# -*- coding: utf-8 -*-


def factorial(n):
    if n == 1:
        return 1
    else:
        return (n) * factorial(n - 1)
