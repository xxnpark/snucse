def f1(n):
    for i in range(1, n+1):
        for j in range(1, i+1):
            print(j, end=" ")
        print()

def f2(n):
    num = 1
    for i in range(n):
        for j in range(i+1):
            print(num, end=" ")
            num += 1
        print()

def f3(n):
    num = 1
    for i in range(n):
        num += i
        for j in range(i+1):
            print(num+j, end=" ")
        print()
    for i in range(n-1, 0, -1):
        num -= i
        for j in range(i):
            print(num+j, end=" ")
        print()

def f4(n):
    num = 1
    for i in range(n):
        for j in range(i+1):
            print(num, end=" ")
            num += 1
        print()
    for i in range(n-1, 0, -1):
        for j in range(i):
            print(num, end=" ")
            num += 1
        print()

def f5(a):
    for i in a:
        sums = 0
        for j in i:
            sums += j
        print(sums)

def f6(a):
    for i in range(len(a)):
        print(a[i][i])

def f7(a):
    b = []
    for i in a:
        sums = 0
        for j in i:
            sums += j
        b.append(sums)
    return b

def f8(a):
    sums = 0
    for i in a:
        for j in i:
            sums += j
    return sums

def f9(a):
    for i in a:
        for j in i:
            if j % 2 == 1:
                print(j, end=" ")
        print()

def f10(a, b):
    for i in range(len(b)):
        for j in range(len(b[i])):
            a[i][j] += b[i][j]
    print(a)

def f11(a, b):
    c = []
    l = len(a)
    m = len(a[0])
    n = len(b[0])
    for i in range(l):
        d = []
        for j in range(n):
            sums = 0
            for k in range(m):
                sums += a[i][k] * b[k][j]
            d.append(sums)
        c.append(d)
    return c

def f12(a):
    ret = True
    for i in range(len(a)):
        for j in range(len(a[i])):
            if i == j and a[i][j] != 1:
                ret = False
            if i != j and a[i][j] != 0:
                ret = False
    return ret

def f13(n, m):
    a = []
    for i in range(n):
        b = []
        for j in range(m):
            b.append(0)
        a.append(b)
    for i in range(n):
        for j in range(m):
            if i != 0:
                a[i-1][j] += 1
            if j != 0:
                a[i][j-1] += 1
            if i != n-1:
                a[i+1][j] += 1
            if j != m-1:
                a[i][j+1] += 1
    return a