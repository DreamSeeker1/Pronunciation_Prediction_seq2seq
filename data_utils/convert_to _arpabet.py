# coding=utf-8
"""Convert the ipa in csv file to arpabet."""

import ipa2arpabet
import csv
import os


def to_csv(filepath_1, filepath_2):
    """read word and ipa from csv file and save the corresponding arpabet to another file.

        Args:
            filepath_1: string, path to the ipa file [word, ipa(several separated by comma)]
            filepath_2: string, path to save the result [word arpabet]
    """

    with open(filepath_1, 'r') as result_csv:
        # use mode 'a' for appending
        with open(filepath_2, 'a') as arpa_result:
            result = csv.reader(result_csv)
            for row in result:
                flag = 0
                word = row[0]
                arpa_list = []
                if len(row) == 2 and row[1] != 'NULL':
                    flag = 1
                    # t is the ipa in corresponding to the word
                    for t in row[1].split(','):
                        if len(t) and (t[-1] == '-' or t[0] == '-'):
                            continue
                        t = t.strip().decode('utf-8')
                        try:
                            arpa_list.append(ipa2arpabet.i2a(t))
                        except ValueError:
                            print t
                            continue
                if flag:
                    for arpa_pron in arpa_list:
                        arpa_result.write(word + ' ')
                        arpa_result.write(' '.join(arpa_pron))
                        arpa_result.write('\n')


def process_result(file_name):
    """Read from the result.csv from webcrawler and convert the ipa to arpabet, save to ./data folder.

        Args:
            file_name: Name of the file for the final conversion result, string.
    """

    data_path = './data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open('../webcrawler/result.csv', 'r') as result:
        # British accent
        en = open(data_path + 'en.csv', 'w')
        # American accent
        us = open(data_path + 'us.csv', 'w')

        en_writer = csv.writer(en)
        us_writer = csv.writer(us)
        result_csv = csv.reader(result)
        for row in result_csv:
            if result_csv.line_num == 1:
                continue
            us_writer.writerow(row[0:2])
            en_writer.writerow([row[0]] + [row[2]])
    to_csv(data_path + 'en.csv', data_path + file_name)
    to_csv(data_path + 'us.csv', data_path + file_name)


process_result("final_result")
