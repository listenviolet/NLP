#coding=utf-8
import sys

def sort_dict(dicts):
	for k, v in dicts.items():
		v = sorted(v)
		dicts[k] = v

def output_dict(key):
	res = dicts[key][0]
	for i in range(1, len(dicts[key]), 1):
		res = res + ',' + dicts[key][i]
	return res 


def Solve(inputs, dicts):
	res = ""
	start = 0

	while start < len(inputs):
		stop = start + 1
		substr = inputs[start: stop]

		while stop <= len(inputs) and substr not in dicts:
			stop += 1
			substr.append(inputs[stop])

		if substr in dicts:
			res += ' ' + substr + '/' + output_dict(substr)
			start = stop
		else:
			res = res + inputs[start]
			start += 1
	return res



if __name__ == '__main__':
	ss = input().split(';')
	
	dicts = {}

	for s in ss:
		s = ss.split('_')
		s[1] = s[1].split('|')

		for i in range(len(s[1])):
			key = s[1][i]
			if key not in dicts:
				dicts[key] = s[0]
			else:
				dicts[key].append(s[0])
	sort_dict(dicts)

	strings = input()
	res = Solve(strings, dicts)
	sys.stdout.write(res + '\n')