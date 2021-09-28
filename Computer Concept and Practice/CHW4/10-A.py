def f1(lst):
    if len(lst) == 0:
        return 0
    else:
        return lst[0] + f1(lst[1:])

def f2(n):
    if n == 1:
        return 1
    elif n%2 == 0:
        return 1+f2(n//2)
    else:
        return 1+f2(n*3+1)

def f3(a):
    if len(a) == 0:
        return
    else:
        f3(a[1:])
        print(a[0])

def f4(lst):
    if not lst:
        return
    if lst[0] % 2:
        print(lst[0] * 3)
    f4(lst[1:])

def f5(lst):
    if lst[-1] % 2 == 1:
        lst[-1] *= 3
    print(lst[-1])
    if len(lst) > 1:
        f5(lst[:-1])
    else:
        return

def f6(lst):
    if not lst:
        return []
    elif isinstance(lst[0], list)==True:
        return f6(lst[0])+f6(lst[1:])
    else:
        return [lst[0]]+f6(lst[1:])
	  
def f7(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1
    elif n > 1:
        return f7(n-1)+f7(n-2)

def f8(string):
    if len(string) <= 1:
        return True
    elif string[0] == string[-1]:
        return f8(string[1:-1])
    else : 
        return False

def f9(n):
    if n == 0:
        return 1
    else:
        return n*f9(n-1)

def f10(lst):
    if not lst:
        return 0
    else:
        return 1+f10(lst[1:])

def f11(a):
    if len(a) == 0:
        return None
    elif len(a) == 1:
        return a[0]
    else:
        return f11(a[1:])

def f12(n):
    if not n: 
        return
    print(n)
    f12(n-1)

def f13(n):
    if n < 10:
        return 1
    else:
        return 1 + f13(n//10)

def f14(lst):
    if lst == []:
        return None
    if lst[0]%2 == 1:
        return lst[0]
    else:
        return f14(lst[1:])

def f15(a):
    if len(a) == 0:
        return 0
    elif a[0]%2 == 1:
        return a[0]+f15(a[1:])
    else:
        return f15(a[1:])

def f16(lst):
    if not lst:
        return []
    elif lst[0] % 2:
        return lst[0:1] + f16(lst[1:])
    else : 
        return f16(lst[1:])

def f17(lst):
    if len(lst) == 2:
        return lst[0]
    else:
        return f17(lst[1:])

def f18(a,b):
    if b == 0:
        return a
    else:
        return f18(b, a%b)

def f19(a, b):
    if len(a) == 0:
        return b
    elif len(b) == 0:
        return a
    elif a[0] < b[0]:
        return a[0:1]+f19(a[1:], b)
    else:
        return b[0:1]+f19(a, b[1:])

def f20(lst):
    if len(lst) <= 1:
        return lst
    return f19(f20(lst[:len(lst)//2]), f20(lst[len(lst)//2:]))