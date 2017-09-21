# -*- coding: utf-8 -*-
'''
提取单词
'''
import re

f = open('./cmudict.0.7a', 'r')
word_list = open('word_list', 'w')

for line in f.readlines():
    t = line.split()[0].lower()
    if re.match('^[a-z]+\'?[a-z]+$', t):
        word_list.write(t + '\n')

f.close()
word_list.close()
