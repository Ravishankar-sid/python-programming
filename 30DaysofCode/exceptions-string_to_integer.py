string = input().strip()

try:
    print(int(string))
except Exception:
    print('Bad String')
