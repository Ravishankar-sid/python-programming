# -*- coding: utf-8 -*-


def solve(meal_cost, tip_percent, tax_percent):
    tip = meal_cost * (tip_percent / 100)
    tax = meal_cost * (tax_percent / 100)

    total_cost = meal_cost + tip + tax

    total_cost = round(total_cost)

    return total_cost
