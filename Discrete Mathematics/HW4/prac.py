def fact(n):
    ret = 1
    for i in range(n):
        ret *= (n+1)
    return ret

print(fact(26) - fact(23) - fact(24) - fact(23) + fact(20))