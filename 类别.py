class dayin(object):
    def __init__(self,i):
        self.i=i
    def printt(self):
        for i in range(1,self.i+1):
            print(i)

a=dayin(10)
a.printt()

class Dayin(object):
    def __init__(self, max_number):
        self.max_number = max_number

    def print_numbers(self):
        for number in range(1, self.max_number + 1):
            print(number)

a = Dayin(10)
a.print_numbers()

class Student(object):
    def __init__(self,student_name,student_age,student_score):
        self.name=student_name
        self.age=student_age
        self.score=student_score
    def print(self):
        print(f"name:{self.name}")
        print(f"age:{self.age}")
        print(f"score:{self.score}")
student1=Student("Alice",20,85)
student1.print()