import re
import numpy as np

f = open('../cmudict.0.7a', 'r')

t_source_list = []
t_target_list = []

for line in f.readlines():
    t = line.split()[0].lower()
    if re.match('^[a-z]+[\'\.]?[a-z.]+$', t):
        t_source_list.append(t)
        t_target_list.append(' '.join(line.split()[1:]))
    elif re.match('^([a-z]+[\'\.]?[a-z.]+)\(2\)$', t):
        tt = re.match('^([a-z]+[\'\.]?[a-z.]+)(\(2\))$', t).group(1)
        t_source_list.append(tt)
        # print line.split()
        t_target_list.append(' '.join(line.split()[1:]))

seq = np.random.permutation(len(t_source_list))

step = 0

training = open('./training', 'w')
testing = open('./testing', 'w')
validation = open('./validation', 'w')
whole = open('./whole', 'w')
data = []

for i in seq:
    data.append((t_source_list[i], t_target_list[i]))


def to_file(data_input, path_name):
    """Write the data to the file
    Args:
        data_input: the input data.
        path_name: the path to save the file.
    """
    # data_input.sort()
    for it in data_input:
        path_name.write(it[0] + ' ' + it[1] + '\n')


t = data[0:10000][:]
assert (len(t) == 10000)
to_file(t, validation)

t = data[10000:20000][:]
assert (len(t) == 10000)
to_file(t, testing)

t = data[20000:][:]
to_file(t, training)

t = data[:]
to_file(t, whole)

validation.close()
testing.close()
training.close()
whole.close()
