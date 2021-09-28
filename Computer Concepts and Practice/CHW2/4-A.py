def f1(a):
    num = 0
    for i in a:
        if i % 2 == 1:
            num += 1
    return num

def f2(a):
    num = 0
    for i in a:
        if i % 2 == 1:
            print(i)

def f3(a):
    sums = 0
    for i in a:
        if i % 2 == 1:
            sums += i
    return sums

def f4(a):
    sums = 0
    for i in range(len(a)):
        if a[i] % 2 == 1:
            sums += i
    return sums

def f5(a):
    b = []
    for i in a:
        b.append(i*i)
    return b

def f6(a):
    maxi = a[0]
    for i in a:
        if maxi < i:
            maxi = i
    return maxi

def f7(a):
    sums = 0
    for i in a:
        sums += i
    return sums/len(a)

def f8(a, b, n):
    for i in range(a, b+1):
        if i % n == 0:
            print(i)

def f9(width, height):
    if width == 0 or height == 0:
        return
    for i in range(height):
        for i in range(width):
            print("*", end="")
        print()

def f10(n):
    for i in range(n):
        for j in range(i+1):
            print("*", end="")
        print()

def f11(a):
    ret = True
    for i in range(len(a)-1):
        if a[i] < a[i+1]:
            ret = False
    return ret

def f12(a):
    ret = True
    for i in a:
        if i >= 0:
            ret = False
    return ret

def f13(a, tar):
    ret = 0
    for i in range(len(a)):
        if a[i] == tar:
            ret = i
    return ret

def f14(a):
    ret = 0
    for i in range(len(a)):
        if a[i] < 0:
            ret = i
    return ret

def f15(a):
    sums = 0
    for i in range(len(a)):
        if i % 2 == 0:
            sums += a[i]
    return sums

def f16(n):
    for i in range(n, 0, -1):
        for j in range(i):
            print("*", end="")
        print()

def f17(a):
    for i in range(len(a)-1, -1, -2):
        print(a[i])

def f18(n):
    fac = 1
    for i in range(1, n+1):
        fac *= i
    return fac

def f19(a):
    for i in a:
        fac = 1
        for j in range(1, i+1):
            fac *= j
        print(fac)

def f20(a):
    for i in a:
        for j in range(i, -1, -1):
            print(j, end=" ")
        print()

def f21(a, b):
    c = []
    for i in a:
        c.append(i)
    for i in range(len(b)):
        c[i] += b[i]
    return c

def f22(n):
    for i in range(1, n+1):
        if i % 2 == 0 or i % 3 == 0:
            print(i)

def f23(a):
    maxi = None
    for i in a:
        for j in i:
            if maxi == None or maxi < j:
                maxi = j
    return maxi

def f24(a):
    fir = a[0]
    sec = a[1]
    if fir < sec:
        fir, sec = sec, fir
    for i in range(2, len(a)):
        if sec < a[i]:
            sec = a[i]
        if fir < sec:
            fir, sec = sec, fir
    return sec

def f25(n):
    while n >= 10:
        n //= 10
    return n

def f26(a):
    for i in a:
        maxi = None
        for j in i:
            if maxi == None or maxi < j:
                maxi = j
        print(maxi)