import numpy as np
import math

def info_gain(examples, attr, label_yes, col):
    #find meadian if its a numberic value
    if(type(examples[attr].iat[0]) is np.int64):
        threshold = np.median(examples[attr])
        gain = entropy(examples, label_yes, col)
        subdata = examples[examples[attr] <= threshold]
        subdata2 = examples[examples[attr] > threshold]
        sub_e = entropy(subdata, label_yes, col)
        sub_e2 = entropy(subdata2, label_yes, col) 
        if(gain == sub_e == sub_e2):
            return gain
        else:
            gain -= (float(len(subdata2)) / float(len(examples))) * sub_e
            gain -= (float(len(subdata)) / float(len(examples))) * sub_e2
        return gain
    else:
 #       print("Not an integer")
        uniq = np.unique(examples[attr])
        gain = entropy(examples, label_yes, col)
        for u in uniq:
            subdata = examples[examples[attr] == u]
            sub_e = entropy(subdata, label_yes, col)
            
            gain -= (float(len(subdata)) / float(len(examples))) * sub_e
            #print(u,sub_e)
        return gain
    
def entropy(examples, label_yes, col):
    pos = 0.0
    neg = 0.0

    for _, row in examples.iterrows():
        if row[col] in label_yes:
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log(p, 2) + n * math.log(n, 2))


def info_gain_Talk(examples, attr, label_yes, col):
    #find meadian if its a numberic value
    if(type(examples[attr].iat[0]) is np.int64):
        threshold = np.median(examples[attr])
        uniq = examples[attr]
        gain = entropy(examples, label_yes, col)
        subdata = examples[examples[attr] <= threshold]
        subdata2 = examples[examples[attr] > threshold]
        sub_e = entropy(subdata, label_yes, col)
        sub_e2 = entropy(subdata2, label_yes, col) 
        print("subdata: ", subdata)
        print("subdata2: ", subdata2)
        if(gain == sub_e == sub_e2):
            return gain
        else:
            gain -= (float(len(subdata2)) / float(len(examples))) * sub_e
            gain -= (float(len(subdata)) / float(len(examples))) * sub_e2
        return gain
    else:
        print("Not an integer")
        uniq = np.unique(examples[attr])
        gain = entropy(examples, label_yes, col)
        print(gain)
        for u in uniq:
            subdata = examples[examples[attr] == u]
            sub_e = entropy(subdata, label_yes, col)
            
            gain -= (float(len(subdata)) / float(len(examples))) * sub_e
            print(u,sub_e)
            print(subdata)
        return gain
    