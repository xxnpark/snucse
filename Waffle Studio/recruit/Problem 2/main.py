import math

'''
def minalba(f, c, x):
    k = 0
    while True:
        if x*(1/(2+k*f) - 1/(2+(k+1)*f)) <= c/(2+k*f):
            return k
        k += 1
'''

# calculate time for specific f, c, x, n
def time(f, c, x, n):
    return sum([c/(2+i*f) for i in range(n)]) + x/(2+n*f)

# data = []
while True:
    try:
        f, c, x = map(int, input().split())
#        n = minalba(f, c, x)
#        data.append((f, c, x, n))

        # n for minimum time
        n = math.ceil(x/c-1-2/f)
        print("%d %.5f" %(n, time(f, c, x, n)))
    except:
        break