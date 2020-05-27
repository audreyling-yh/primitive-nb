import csv
import random
import unicodedata
import string

#################################################################################
# import data
#################################################################################
def read_data(filepath):
    # get the data in a list
    raw_data = []
    with open(filepath, newline='', encoding = 'utf-8-sig') as csvfile:
        raw = csv.reader(csvfile)
        for row in raw:
            raw_data.append(row)
    return raw_data

raw = read_data('../data/name_nationality.csv')
variables = raw[0]
data = raw[1:]

#################################################################################
# clean data
#################################################################################
def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)

for x in data:
    name = x[0]
    name = strip_accents(name)
    name = name.replace(' ', '')
    name = ''.join([i if i.isalpha() else '' for i in name]).lower()
    name = '_{}_'.format(name)
    x.append(name)

variables.append('name_clean')

#################################################################################
# feature engineering
#################################################################################
def check_alphabet(name, alphabet):
    if alphabet in name:
        return 1
    else:
        return 0

feature_list = []
alphabet_string = list(string.ascii_lowercase) + ['_']
for i in alphabet_string:
    for j in alphabet_string:
        feature = i + j
        feature_list.append(feature)

for x in feature_list:
    variables.append(x)

for x in data:
    name_clean = x[2]
    for i in feature_list:
        res = check_alphabet(name_clean, i)
        x.append(res)

#################################################################################
# train test split
#################################################################################
korean = [x for x in data if x[1] == 'korean']
japanese = [x for x in data if x[1] == 'japanese']
chinese = [x for x in data if x[1] == 'chinese']
random.seed(42)
random.shuffle(korean)
random.shuffle(japanese)
random.shuffle(chinese)

training = []
test = []
split_idx = int(len(korean)*0.7)
for x in korean[:split_idx]:
    training.append(x)
for x in japanese[:split_idx]:
    training.append(x)
for x in chinese[:split_idx]:
    training.append(x)
for x in korean[split_idx:]:
    test.append(x)
for x in japanese[split_idx:]:
    test.append(x)
for x in chinese[split_idx:]:
    test.append(x)

#################################################################################
# train the model
#################################################################################
outcomes = ['korean', 'japanese', 'chinese']

count_dict = {}
for o in outcomes:
    var = '_'.join(['count', o])
    count_dict[var] = len(training)/3
    idx = 3
    for i in feature_list:
        for j in [0, 1]:
            count = 0
            var = '_'.join(['count', o, i, str(j)])
            for x in training:
                if x[1] == o:
                    if x[idx] == j:
                        count += 1
            count_dict[var] = count
        idx += 1

# prior and likelihood
prob_dict = {}
for o in outcomes:
    count_o_var = '_'.join(['count', o])
    p_o_var = '_'.join(['p', o])
    prob_dict[p_o_var] = count_dict[count_o_var] / len(training)

    for i in feature_list:
        for j in [0, 1]:
            count_var = '_'.join(['count', o, i, str(j)])
            p_var = '_'.join(['p', o, i, str(j)])
            prob_dict[p_var] = count_dict[count_var] / count_dict[count_o_var]

#################################################################################
# get test predictions
#################################################################################
for x in test:
    pred_dict = {}
    for o in outcomes:
        var = '_'.join(['p', o])
        pred_dict[var] = prob_dict[var]
        idx = 3
        for i in feature_list:
            category = x[idx]
            p_name = '_'.join(['p', o, i, str(category)])
            pred_dict[var] = pred_dict[var] * prob_dict[p_name]
            idx += 1
    highest_p = max(pred_dict, key = pred_dict.get)
    pred = highest_p.strip('p_')
    x.append(pred)

variables.append('predicted_nationality')

#################################################################################
# get performance measures
#################################################################################
def calc(num, den, tp, fp, fn):
    try:
        metric = num/den
        return metric
    except ZeroDivisionError:
        if tp == fp == fn == 0:
            return 1
        elif tp == 0 and (fp != 0 or fn != 0):
            return 0

perf_measures = []
for o in outcomes:
    prec_predicted, prec_correct, recall_actual, recall_correct, TP, TN, FP, FN = ((0,) * 8)
    for i in test:
        actual = i[variables.index('nationality')]
        pred = i[variables.index('predicted_nationality')]

        if pred == o:
            prec_predicted += 1
            if actual == o:
                prec_correct += 1
        if actual == o:
            recall_actual += 1
            if pred == o:
                recall_correct += 1
                TP += 1
            elif pred != o:
                FN += 1
        elif actual != o:
            if pred != o:
                TN += 1
            elif pred == o:
                FP += 1
    precision = calc(prec_correct, prec_predicted, TP, FP, FN)
    recall = calc(recall_correct, recall_actual, TP, FP, FN)
    f = calc((2 * precision * recall), (precision + recall), TP, FP, FN)
    acc = (TP + TN) / (TP + FP + FN + TN)

    prec_name = '_'.join(['precision', o])
    recall_name = '_'.join(['recall', o])
    f_name = '_'.join(['f', o])
    acc_name = '_'.join(['accuracy', o])
    perf_measures.append([prec_name, precision])
    perf_measures.append([recall_name, recall])
    perf_measures.append([f_name, f])
    perf_measures.append([acc_name, acc])

for x in perf_measures:
    if 'accuracy' in x[0]:
        print('{}: {}'.format(x[0], round(x[1], 3)))
