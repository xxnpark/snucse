# Simple OOP Project (1) : Animal and Dog

class Animal:
    def __init__(self):
        print("Animal created")

    def whoAmI(self):
        print("Animal")

    def eat(self):
        print("Eating")

class Dog(Animal):
    def __init__(self):
        super().__init__()
        print("Dog created")

    def whoAmI(self):
        print("Dog")

    def bark(self):
        print("Woof!")

d = Dog()
d.whoAmI()
d.eat()
d.bark()

print("\n" + "-"*30 + "\n")
# Simple OOP Project (2) : Circle

class Circle():
    def __init__(self):
        self.pi = 3.141592
        self.radius = 0

    def setRadius(self, r):
        self.radius = r

    def getRadius(self):
        return self.radius

    def area(self):
        return self.pi * self.radius**2

c = Circle()
c.setRadius(5)
print(c.getRadius())
print(c.area())

print("\n" + "-"*30 + "\n")
# Simple OOP Project (3) : Shape and Others

class Shape:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.description = "This shape has not been described yet"
        self.author = "Nobody has claimed to make this shape yet"

    def area(self):
        return self.x * self.y

    def perimeter(self):
        return 2 * self.x + 2 * self.y

    def describe(self, text):
        self.description = text

    def authorName(self, text):
        self.author = text

    def scaleSize(self, scale):
        self.x = self.x * scale
        self.y = self.y * scale

# Simple OOP Project (3) : Shape and Others [1/3]

rectangle = Shape(100, 45)
print(rectangle.area())
print(rectangle.perimeter())
rectangle.describe("A wide rectangle, more than twice\
 as wide as it is tall")
rectangle.scaleSize(0.5)
print(rectangle.area())

# Simple OOP Project (3) : Shape and Others [2/3]

class Square(Shape):
    def __init__(self, a):
        super().__init__(a, a)

class DoubleSquare(Square):    
    def perimeter(self):
        return 4 * self.x + 2 * self.y

    def area(self):
        return 2 * self.x * self.y

# Simple OOP Project (3) : Shape and Others [3/3]

class InsideDoubleSquare(Square):
    def perimeter(self):
        return self.x + self.y

    def area(self):
        return self.x * self.y / 4.0