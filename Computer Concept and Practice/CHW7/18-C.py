#18-C 코드
# 1번
def match_func(lst, value):
	x = 0
	if len(lst) == 0 :
		return 0
	elif lst[0] == value:
		x = x+1
	return x + match_func(lst[1:], value)

# 2번
def twice_elem(lst):
	newlist = []
	if lst == []:
		return newlist
	else:
		newlist.append(lst[0])
		newlist.append(lst[0])
		return newlist + twice_elem(lst[1:])

# 3번
def check_sum(nums,k):
	if len(nums) == 0:
		return bool(0==k)
	else:
		k = k - nums[0]
		return check_sum(nums[1:],k)

# 4번
def repeat_elem(L):
	chk = []
	for i in L:
		if i not in chk:
			chk.append(i)
	for i in chk:
		L.remove(i)
	s = set(L)
	return sorted(s)