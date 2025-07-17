import os

words = []
with open('disease/total.csv','r') as f:
    lines = f.readlines()
    for line in lines:
        words.append(line.strip())
words = list(set(words))
print(len(words))

with open('disease/total_norepeat.csv','w') as f:
    for word in words:
        f.write(word+'\n')

