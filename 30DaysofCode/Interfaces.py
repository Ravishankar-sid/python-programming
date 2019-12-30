class Calculator(AdvancedArithmetic):
    def divisorSum(self, n):
        if n == 1:
            return 1
        else:
            _sum = 1 + n 
            for i in range(2, n//2 + 1):
                if n % i == 0:
                    _sum += i
            return _sum
