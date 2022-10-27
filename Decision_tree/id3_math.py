import numpy as np
import math

def info_gain(examples, attr, label_yes, col):
    uniq = np.unique(examples[attr])
    gain = entropy(examples, label_yes, col)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata, label_yes, col)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
       # print(u,sub_e)
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

def info_gain_w(examples, attr, label_yes, col,w):
    #find meadian if its a numberic value
    #print("examples[attr]: ",examples[attr][0])
    if(type(examples[attr][0]) == int):
        threshold = np.median(examples[attr])
        uniq = examples[attr]
        gain = entropy_w(examples, label_yes, col,w)
        subdata = examples[examples[attr] < threshold]
        subdata2 = examples[examples[attr] > threshold]

        sub_e = entropy_w(subdata, label_yes, col,w)
        sub_e2 = entropy_w(subdata2, label_yes, col,w) 
        gain -= (float(len(subdata2)) / float(len(examples))) * sub_e
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e2
            #print(u,sub_e)
        return gain
    else:
        uniq = np.unique(examples[attr])
        gain = entropy_w(examples, label_yes, col,w)
        for u in uniq:
            subdata = examples[examples[attr] == u]
            sub_e = entropy_w(subdata, label_yes, col,w)
            
            gain -= (float(len(subdata)) / float(len(examples))) * sub_e
            #print(u,sub_e)
        return gain
    
def entropy_w(examples, label_yes, col,w):
    pos = 0.0
    neg = 0.0
    for i, row in examples.iterrows():
        if row[col] in label_yes:
            pos += w[i]
        else:
            neg += w[i]
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log(p, 2) + n * math.log(n, 2))


def gain_ME(examples, attr, label_yes, col):
    uniq = np.unique(examples[attr])
    set_length = examples.shape[0]
    gain = ME(examples, label_yes, col)
    #print("Total gain:", gain)
    for u in uniq:
        
        subdata = examples[examples[attr] == u]
        sub_e = ME(subdata, label_yes, col)
        #print("Subdata:", subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * (sub_e/set_length)
    return gain

def ME(examples, label_yes, col):
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
        return neg


def gain_GI(examples, attr, label_yes, col):
    uniq = np.unique(examples[attr])
    gain = GI(examples, label_yes, col)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = GI(subdata, label_yes, col)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
    return gain
def GI(examples, label_yes, col):
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
        return 1-n**2 - p**2
