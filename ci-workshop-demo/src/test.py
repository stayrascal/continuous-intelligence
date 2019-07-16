


str1 = 'ukq pdkqcdp e swjpaz pk owu pdwp e hkra ukq, xqp e swjp pk owu, bqyg ukq!'
str2 = 'ulrl dcofbofykx xwprxfis zuhibx mityn if vbzlla dlcka qi pexciukx,'
out =  'your grilfriend actually called power li before going to Thailand'
key =  'wxxu xuxuwxxuxu xuwxxuxu xuwxxu xuxuw xx uxuxuw xxuxu xu wxxuxuxuwxxuxuxu'
'wxxuxuxu'

def tranlate(input, len=0):
    # print(input)
    out = ''
    for i in input:
        if i == ',' or i == ' ' or i == '!':
            out = out + i
        else:
            index = ord(i) + len
            if index > 122:
                index = index - 26
            out = out + chr(index)
            
    return out

# print(tranlate(str1, 4))

for i in range(1, 27):
    print(tranlate(str2, i))
# print(ord('a'))
# print(ord('z'))

# print(str2)
# print(tranlate(str2,0))