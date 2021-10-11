# input
n, p, k = map(int, input().split())
stu = dict()
for r in range(n):
    temp = list(map(int, input().split()))
    stu[temp[0]] = temp[1:]
ans = list(map(int, input().split()))

# act
scores = dict()
for i in range(p+1) : scores[i] = []
result = dict()
for i in stu:
    score = 0
    for j in range(p):
        if stu[i][j] == ans[j] : score += 1 # calculate score
    scores[score].append(i) # scores >>> dictionary(score : [codes])
    result[i] = score # result >>> dictionary(code : score)
for i in range(p, 0, -1):
    if i == p : scores[p+1] = 1
    scores[i] = len(scores[i]) + scores[i+1] # scores >>> dictionary(score : rank)
result = sorted(result.items()) # sort dictionary by keys

# print
for i in result : print("Student #%d: %d %d" %(i[0], i[1], scores[i[1]+1]))