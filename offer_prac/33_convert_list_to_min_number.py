from functools import cmp_to_key
def compare(str1, str2):
	comb1 = str1
	comb1 += str2 # str1,str2

	comb2 = str2
	comb2 += str1 # str2,str1

	if comb1 < comb2: return -1
	if comb1 == comb2: return 0
	return 1

def PrintMinNUmber(numbers):
	if not numbers: return ""
	numbers = [str(number) for number in numbers]
	numbers = sorted(numbers, key = cmp_to_key(compare))
	res = ""
	for number in numbers:
		res += number
	return res
numbers = [4,3,32,321]
print(PrintMinNUmber(numbers))

# Alg:
# 1. define a new compare rule
# 2. Proof: the compare rule 
#    reflexivity
#    symmetry
#    transitivity
# 3. Proof: such sort function can find the min number

# Python notice:
# in python3, to customize the compare function:
# from functools import cmp_to_key
# sorted_arr = sorted(arr, key = cmp_to_key(self_def_cmp_func))
# sorted(): not an inplace function