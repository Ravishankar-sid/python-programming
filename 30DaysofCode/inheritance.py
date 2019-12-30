class Student(Person):
    #   Class Constructor
    #   
    #   Parameters:
    #   firstName - A string denoting the Person's first name.
    #   lastName - A string denoting the Person's last name.
    #   id - An integer denoting the Person's ID number.
    #   scores - An array of integers denoting the Person's test scores.
    #
    # Write your constructor here
    def __init__(self, firstName, lastName, idNumber, scores):
        super().__init__(firstName, lastName, idNumber)
        self.scores = scores
    #   Function Name: calculate
    #   Return: A character denoting the grade.
    #
    # Write your function here
    def calculate(self):    
        _average=sum(self.scores)//len(self.scores)
        if(_average >=90 and _average <=100 ):
            return("O")
        elif(_average >= 80 and _average < 90):
            return("E") 
        elif(_average >= 70 and _average < 80):
            return("A")
        elif(_average >= 55 and _average < 70):
            return("P")
        elif(_average >= 40 and _average < 55):
            return("D")
        elif(_average < 40):
            return("T")   
