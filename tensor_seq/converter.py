# -*- coding: utf-8 -*-
'''
提取单词
'''

import re

f = open('../cmudict.0.7a', 'r')
source_list = open('source_list', 'w')
target_list = open('target_list', 'w')
for line in f.readlines():
    t = line.split()[0].lower()
    if re.match('^[a-z]+\'?[a-z]+$', t):
        source_list.write(t + '\n')
        # print line.split()
        target_list.write(' '.join(line.split()[1:]) + '\n')
f.close()
source_list.close()
target_list.close()
