import sys
def inprocess(strings):
    dicts = {'<':'>', '>':'<', '[':']', ']':'[' }

    flag = []
    i = 0
    res = []
    start = 0
    # split inputs to many strings
    while i <len(strings):
        s = strings[i]
        # print("s = ", s)
        if s=='<' or s=='[':
            flag.append(s)

            if strings[start]=='@':
                res.append(strings[start:i])
                start = i  

        elif flag and s==dicts[flag[-1]]:
            # print("s==dicts[flag[-1]] = ", s)
            tmp = flag.pop()
            if flag:
                continue
            else:
                res.append(strings[start+1:i])
                start = i+1
            del tmp

        i += 1
        # print("--------")

    # process the res
    collect = []

    for id, lists in enumerate(res):
        print(lists)
        if '|' not in lists:
            collect.append(lists)
            continue

        lists = lists.split('|')
        for s in lists:
            
            i = 0
            # while i < len(s):
            if s[i]=='[':
                collect.append(s[i+1:s.index(']')])
                i = s.index(']')+1
                if i<len(s):
                    collect.append(collect[-1]+s[i:])

            elif '[' not in s:
                collect.append(s)
    # for c in res:
    #    print(c)
    print("collect:")
    for id, lists in enumerate(collect):
        print(lists)
    return collect

def testing(qurry, collect):
    i = 0
    # last = 0
    while i<len(qurry):
        c = qurry[i]
        if c in collect:
            collect = collect[collect.index(c)+1:]
        elif c=='@':
            if qurry[i:qurry.index('}')+2] in collect:
                i = qurry.index('}') + 1
            else:
                return 0
        else:
            return 0
        i += 1
    return 1

def testing(query, collect):
    start = 0
    last = 0;
    while start < len(query):
        

inputs = raw_input()
query = raw_input()

# inputs = input()
# query = input()

res = inprocess(inputs)
# sys.stdout = testing(query, res)
sys.stdout.write(str(testing(query, res)) + '\n')
# print(testing(query,res))