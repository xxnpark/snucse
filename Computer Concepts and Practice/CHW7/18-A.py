# Advanced OOP 1번
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Pythagoras():
    def __init__(self):
        self.point_one = Point(0, 0)
        self.point_two = Point(0, 0)

    def setPointOne(self, point1):
        self.point_one = point1

    def setPointTwo(self, point2):
        self.point_two = point2

    def getSlope(self):
        if self.point_one.x == self.point_two.x:
            return None
        else:
            return (self.point_two.y-self.point_one.y)/(self.point_two.x-self.point_one.x)

    def getDistance(self):
        return ((self.point_two.x-self.point_one.x)**2+(self.point_two.y-self.point_one.y)**2)**0.5


# Advanced OOP 2번
class Calculator():
    def __init__(self):
        self.num = 0
        self.current = ""
        self.history = []
    
    def add(self, int):
        self.num += int
        if not self.current: self.current += str(int)
        else : self.current += " + " + str(int)
    
    def subtract(self, int):
        self.num -= int
        self.current += " - " + str(int)

    def multiply(self, int):
        self.num *= int
        self.current += " * " + str(int)
        
    def equals(self, boolean=False):
        if not self.current: print("No calculation done yet!")
        else:
            self.current += " = " + str(self.num)
            self.history.append(self.current)
            if boolean:
                print(self.num)
            self.num = 0
            self.current = ""

    def showHistory(self):
        print("History:")
        for eq in self.history : print(eq)

test = Calculator()
test.equals()
test.showHistory()

test.add(2)
test.subtract(1)
test.equals()
test.showHistory()

test.add(2)
test.multiply(4)
test.equals(True)

test.add(10)
test.subtract(5)
test.multiply(2)
test.equals()

test.showHistory()


# Advanced OOP 3번
class Account:
    def __init__(self,account_holder):
        self.balance = 0
        self.holder = account_holder
        self.transactions = []
    
    def deposit(self,amount):
        self.balance += amount
        self.transactions.append(('deposit',amount))
    
    def withdrawal(self,amount):
        if amount>self.balance:
            return "Insufficient funds"
        self.balance -= amount
        self.transactions.append(('withdrawal',amount))
    
    def status(self):
        print(self.holder+": ",end="")
        print(self.transactions)

bob_account = Account('Bob')
bob_account.deposit(1000000)
bob_account.withdrawal(100)
bob_account.deposit(440)
bob_account.status()

tom_account = Account('Tom')
tom_account.deposit(5000000)
tom_account.withdrawal(250)
tom_account.withdrawal(875)
tom_account.status()


# Advanced OOP 4번
class atom:
    Atno_to_Symbol = {1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O'}
    def __init__(self,atno,x,y,z):
        self.atno = atno
        self.position = (x,y,z)
    def symbol(self):
        return self.Atno_to_Symbol[self.atno]
    def __repr__(self):
        return '%d %10.4f %10.4f %10.4f'%(self.atno,self.position[0],self.position[1],self.position[2])

class molecule:
    def __init__(self,name='Generic'):
        self.name = name
        self.atomlist = []
    def addatom(self,atom):
        self.atomlist.append(atom)
    def __repr__(self):
        str = 'This is a molecule named %s\n' %self.name
        str += 'It has %d atoms\n' %len(self.atomlist)
        for atom in self.atomlist:
            str += ' %s \n'%atom
        return str

at = atom(6, 0.0, 1.0, 2.0)
print(at)
print(at.symbol())

mol = molecule('Water')
at = atom(8,0.,0.,0.)
mol.addatom(at)
mol.addatom(atom(1,0.,0.,1.))
mol.addatom(atom(1,0.,1.,0.))
print(mol)


# Advanced OOP 5번
class Person():
    def __init__(self, name, depart):
        self.name = name
        self.depart = depart
    
    def getName(self):
        return self.name

    def getDepart(self):
        return self.depart

class Student(Person):
    def __init__(self, name, depart, year, credit):
        super().__init__(name, depart)
        self.year = year
        self.credit = credit
    
    def setCredit(self, credit):
        self.credit = credit

    def getCredit(self):
        return self.credit

    def increaseYear(self):
        self.year += 1

class Professor(Person):
    def __init__(self, name, depart, course, salary):
        super().__init__(name, depart)
        self.course = course
        self.salary = salary
    
    def getCourse(self):
        return self.course
    
    def getAnnualSalary(self):
        return self.salary * 12

    def raiseSalary(self, percent):
        self.salary *= 1 + percent / 100

tim_cook = Professor("Tim Cook", "CSE", "Soft. Arch.", 5500)
print("sum of 5 year annual salary:", tim_cook.getAnnualSalary() * 5)
sum = 0
for i in range(5):
    sum += tim_cook.getAnnualSalary()
    tim_cook.raiseSalary(15)
print("sum of 5 year annual salary with 15% increase:", sum)


# Advanced OOP 6번
class Person():
    def __init__(self, name, address):
        self.name = name
        self.address = address

    def getName(self):
        return self.name
    
    def getAddress(self):
        return self.address

class Student(Person):
    def __init__(self, name, address, gpa, year, fee):
        super().__init__(name, address)
        self.gpa = gpa
        self.year = year
        self.fee = fee

    def getGpa(self):
        return self.gpa

    def setGpa(self, gpa):
        self.gpa = gpa

    def hasMinimumGpa(self):
        return bool(self.gpa >= 3.5)
    
    def willGraduateNextYear(self):
        return bool(self.year == 4)

class Staff(Person):
    def __init__(self, name, address, school, annual_pay):
        super().__init__(name, address)
        self.school = school
        self.annual_pay = annual_pay

    def getSchool(self):
        return self.school

    def getMonthlyPay(self):
        return self.annual_pay/12

    def raiseAnnualPay(self, percent):
        self.annual_pay *= 1+percent/100

tom = Staff("Tom", "Gangnam", "Yonsei", 35000)
dane = Staff("Dane", "Shindorim", "Sogang", 20000)

for _ in range(7):
    tom.raiseAnnualPay(7)
    dane.raiseAnnualPay(15)

if tom.getMonthlyPay() > dane.getMonthlyPay():
    print("Tom has a larger monthly pay")
else:
    print("Dane has a larger monthly pay")