''' 把只包含质因子2、3和5的数称作丑数（Ugly Number）。
	例如6、8都是丑数，但14不是，因为它包含质因子7。 
	习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
'''

''' Alg1: exceeds time limit
'''
def isUglyNumber(number):
	factors = [2, 3, 5]
	flag = 0
	for factor in factors:
		while number % factor == 0: 
			number = number // factor
		if number == 1: 
			flag = 1
			break
	if flag == 1: return True
	else: return False

def GetUglyNumber_Solution(index):
	if index < 1: return 0
	cnt = 0
	number = 0
	res = []
	while cnt < index:
		number += 1
		if isUglyNumber(number):
			res.append(number)
			cnt += 1
	return res[-1]

print(GetUglyNumber_Solution(0))

'''Alg2
'''

def GetUglyNumber_Solution2(index):
	if index < 1: return 0
	ugly_num_list = [1]
	while len(ugly_num_list) < index:
		next_ugly_num = min(nextUglyNumberForFactor(ugly_num_list, 2),
							nextUglyNumberForFactor(ugly_num_list, 3),
							nextUglyNumberForFactor(ugly_num_list, 5))
		ugly_num_list.append(next_ugly_num)
	return ugly_num_list[-1]

def nextUglyNumberForFactor(ugly_num_list, factor):
	cur_max = ugly_num_list[-1]
	mid = cur_max // factor
	for i in range(len(ugly_num_list)):
		if ugly_num_list[i] > mid:
			return ugly_num_list[i] * factor


print(GetUglyNumber_Solution2(0))

# Notice: 
# input:  0
# output: 0

