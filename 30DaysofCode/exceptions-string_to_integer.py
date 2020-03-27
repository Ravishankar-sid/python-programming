# -*- coding: utf-8 -*-

string = input().strip()

try:
    print(int(string))
except Exception:
    print("Bad String")
