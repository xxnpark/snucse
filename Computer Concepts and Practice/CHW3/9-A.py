def f1(a):
    return len([0 for x in a if x%2])

def f2(a):
    [print(x) for x in a if x%2]

def f3(a):
    return sum([x for x in a if x%2])

def f4(a):
    return sum([i for i, x in enumerate(a) if x%2])

def f5(a):
    return [x**2 for x in a]

def f6(a):
    return max(a)

def f7(a):
    return sum(a)/len(a)

def f8(a, b, n):
    [print(x) for x in range(((a-1)//n)*n+n, b+1, n)]

def f9(w, h):
    [print("*"*w) for i in range(h) if w]

def f10(n):
    [print("*"*i) for i in range(1, n+1)]

def f11a(a):
    return a == sorted(a, reverse=True)

def f11b(a):
    return all([a.pop() <= a.copy().pop() for i in range(len(a)-1)])

def f12(a):
    return all([x < 0 for x in a])

def f13(a, tar):
    return max([i for i, x in enumerate(a) if x == tar])

def f14a(a):
    return max(map(lambda i: i if a[i] < 0 else 0, range(len(a))))

def f14b(a):
    return max([i for i, x in enumerate(a) if x < 0])

def f15(a):
    return sum(a[::2])

def f16(n):
    [print("*"*(n-i)) for i in range(n)]

def f17(a):
    [print(x) for x in a[::-2]]

def f18(n):
    return n*f18(n-1) if n > 0 else 1

def f19(a):
    [print(sum(b)) for b in a]

def f20(a):
    [print(x) for x in [y for b in a for y in b][slice(0, len(a)*len(a), len(a)+1)]]

def f21(a):
    [print(eval("*".join([str(i) for i in range(1, n+1)]))) if n > 0 else print(1) for n in a]

def f22(lst):
    [print(' '.join([str(i) for i in range(n, -1, -1)])) for n in lst]

def f23(a, b):
    return [i+j for i, j in zip(a, b)]

def f24(n):
    [print(i) for i in range(1, n+1) if i%2 == 0 or i%3 == 0]

def f25a(a):
    return max([max(b) for b in a if b != []])

def f25b(a):
    return max([x for b in a for x in b])

def f26(a):
    return sorted(a)[-2]

def f27(n):
    return int(str(n)[0])

def f28(a):
    [print(max(b)) for b in a]