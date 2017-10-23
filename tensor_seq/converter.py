# -*- coding: utf-8 -*-
'''
提取单词
'''
import os


def extract_words(dict_path, source_path, target_path, file_name):
    """Read the content from the dictionary and split the source
       and target sequence, then write to file.

       Args:
           dict_path: path to the dictionary file
           source_path: path to save the source sequence
           target_path: path to save the target sequence
           file_name: file name in the corresponding path
    """
    dict_path, source_path, target_path = dict_path + file_name, source_path + file_name, target_path + file_name
    f = open(dict_path, 'r')
    source_list = open(source_path, 'w')
    target_list = open(target_path, 'w')

    for line in f.readlines():
        t = line.split()[0].lower()
        source_list.write(t + '\n')
        target_list.write(' '.join(line.split()[1:]) + '\n')
    f.close()
    source_list.close()
    target_list.close()


data_set_path = './dataset/'
if not os.path.exists(data_set_path):
    os.makedirs(data_set_path)

dict_path_pre = '../Split_Dataset/'
source_path_pre = data_set_path + 'source_list_'
target_path_pre = data_set_path + 'target_list_'
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'training')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'testing')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'validation')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'whole')