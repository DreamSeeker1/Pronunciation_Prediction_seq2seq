# coding=utf-8
"""Convert the ipa in csv file to arpabet."""

import ipa2arpabet
import csv


def to_csv(filepath_1, filepath_2):
    """read word and ipa from csv file and save the corresponding arpabet to another file
        Args:
            filepath_1:string, path to the ipa file [word, ipa(several separated by comma)]
            filepath_2:string, path to save the result [word arpabet]
    """

    with open(filepath_1, 'r') as result_csv:
        with open(filepath_2, 'w') as arpa_result:
            result = csv.reader(result_csv)
            for row in result:
                flag = 0
                word = row[0]
                arpa_list = []
                if row[1] != 'NULL':
                    flag = 1
                    # t is the ipa in corresponding to the word
                    for t in row[1].split(','):
                        if len(t) and t[-1] == '-':
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
