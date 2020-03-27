# -*- coding: utf-8 -*-

import re

list_of_names = []

for i in range(int(input())):
    first_name, emails = [str(string) for string in input().split()]
    if re.search("@gmail\.com$", emails):
        list_of_names.append(first_name)

print(*sorted(list_of_names), sep="\n")
