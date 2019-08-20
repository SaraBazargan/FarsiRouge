import codecs
import re
fp = open('C://Users/sara/Desktop/rouge implementation/Fa_data.txt', 'r', encoding= 'utf-8')
txt = fp.readlines()
count = 0
sen = 0
for line in txt:
    #print(line)
    length = []
    length = len(line.split())
    #print(length)
    count += length
    if length != 0:
        sen += 1
    
print(count, count/sen)




fp = open('C://Users/sara/Desktop/rouge implementation/kholaseha/model/model.D.001.txt', 'r', encoding= 'utf-8')
txt = fp.readlines()
count = 0
sen = 0
for line in txt:
    #print(line)
    length = []
    length = len(line.split())
    #print(length)
    count += length
    if length != 0:
        sen += 1
    
print(sen, count, count/sen)
