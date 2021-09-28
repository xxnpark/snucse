#18-B 코드
# 1번
def first_perfect_square(numbers):
	lst = []
	import math
	for i in range(len(numbers)):
		if numbers[i] >= 0:
			k = numbers[i]
			j = int(math.sqrt(k))
			if int(j**2) == k:
				lst.append(i)
	if len(lst) == 0:
		return -1
	else:
		return lst[0]

# 2번
def num_perfect_squares(numbers):
	import math
	countt = 0
	for i in range(len(numbers)):
		if numbers[i] >= 0:
			k = numbers[i]
			j = int(math.sqrt(k))
			if int(j**2) == k :
				countt = countt + 1
	return countt

#3번
def second_largest(values):
	values.remove(max(values))
	return (max(values))
