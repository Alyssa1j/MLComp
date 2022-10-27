
import numpy as np

import tree_node as tree
import id3_math as m


def ID3(data, attrs, label_yes, label_no,answer, depth=6):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.info_gain(data, attrs[feature], label_yes, answer)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = (max_feat, attrs[max_feat])
    root.level=depth
    root.infogain = gain
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]
        result = m.entropy(subdata, label_yes, answer)
        if  result == 0.0 or (depth-1)==1:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = (u,answer)   
            newNode.pred = np.unique(subdata[answer])
            if(any(p in newNode.pred for p in label_no) and any(p in newNode.pred for p in label_yes)):
                newNode.pred = subdata[answer].value_counts().idxmax()

           # print(newNode.value, newNode.pred)
            newNode.entropy = result
            newNode.level = depth-1  
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = (u, attrs[max_feat])
            #print(dummyNode.value)
            dummyNode.level=depth-1
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)      
            child = ID3(subdata, new_attrs, label_yes, answer,depth-2)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

#id3 with weighted samples:
def ID3_weighted(data, attrs, label_yes, label_no,answer, weights, depth=6):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.info_gain_w(data, attrs[feature], label_yes, answer, weights)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = (max_feat, attrs[max_feat])
    root.level=depth
    root.infogain = gain
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]
        result = m.entropy_w(subdata, label_yes, answer, weights)
        if  result == 0.0 or (depth-1)==1:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = (u,answer)   
            newNode.pred = np.unique(subdata[answer])
            if(any(p in newNode.pred for p in label_no) and any(p in newNode.pred for p in label_yes)):
                newNode.pred = subdata[answer].value_counts().idxmax()

           # print(newNode.value, newNode.pred)
            newNode.entropy = result
            newNode.level = depth-1  
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = (u, attrs[max_feat])
            #print(dummyNode.value)
            dummyNode.level=depth-1
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)      
            child = ID3_weighted(subdata, new_attrs, label_yes, answer,weights,depth-2)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

#id3 with weighted samples and binary threshold
def ID3_W_BT(data, attrs, label_yes, label_no,answer, weights, depth=6):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.info_gain_w(data, attrs[feature], label_yes, answer, weights)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = (max_feat, attrs[max_feat])
    root.level=depth
    root.infogain = max_gain
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]
        result = m.entropy_w(subdata, label_yes, answer, weights)
        if  result == 0.0 or (depth-1)==1:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = (u,answer)   
            newNode.pred = np.unique(subdata[answer])
            if(any(p in newNode.pred for p in label_no) and any(p in newNode.pred for p in label_yes)):
                newNode.pred = subdata[answer].value_counts().idxmax()

           # print(newNode.value, newNode.pred)
            newNode.entropy = result
            newNode.level = depth-1  
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = (u, attrs[max_feat])
            #print(dummyNode.value)
            dummyNode.level=depth-1
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)      
            child = ID3_weighted(subdata, new_attrs, label_yes, answer,weights,depth-2)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

def ID3_ME(data, attrs, label_yes, label_no, answer, depth=0):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.gain_ME(data, attrs[feature], label_yes, label_no, answer)
        #print(feature+":")
       # print(gain)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]

        if m.ME(subdata, label_yes, answer) == 0.0:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata[answer])
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)
            child = ID3_ME(subdata, new_attrs, label_yes, label_no, answer)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

def ID3_GI(data,attrs, label_yes, label_no, answer, depth=0):
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.gain_GI(data, attrs[feature], label_yes, label_no, answer)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root.value = max_feat
    uniq = np.unique(data[attrs[max_feat]])
    for u in uniq:
        subdata = data[data[attrs[max_feat]] == u]

        if m.GI(subdata, label_yes, answer) == 0.0:
            newNode = tree.Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata[answer])
            root.children.append(newNode)
        else:
            dummyNode = tree.Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.pop(max_feat)
            child = ID3_ME(subdata, new_attrs, label_yes, label_no, answer)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

def ID3_Method(method,data,attrs, label_yes, answer, depth=6):
    if method == "ME":
        return ID3_ME(data,attrs, label_yes, label_no, answer, depth)
    if method == "GI":
        return ID3_GI(data,attrs, label_yes, label_no, answer, depth)
    else:
        return ID3(data,attrs, label_yes, answer, depth)

def printTree(root, depth=0):
        for i in range(depth):
            print("\t", end="")
        print(root.value,root.level, root.infogain, end="")
        if root.isLeaf:
            print(" -> ", root.pred, root.entropy)
        print()
        for child in root.children:
            printTree(child, depth + 1)

def prediction_result(root, dataset, features,label_yes,label):
    rows,colms = dataset.shape
    correct =0
    for _,data_row in dataset.iterrows():
      #  print("data_row:",data_row)
        b = prediction(root, data_row, features, label_yes,label)
        if b:
            correct+=1
    return correct/rows

def prediction(root, data_row, features,label_yes,label):
    if root.isLeaf:
        if data_row[label] in root.pred:
            return True
        else:
            return False
    else:
        for c in root.children:
            #compare values then pick the path that fits.
            if(c.value[0] == data_row[features[root.value[0]]]):
                if(c.isLeaf):
                    return prediction(c, data_row, features, label_yes,label)
                else:
                    return prediction(c.children[0], data_row, features, label_yes,label)
            else:
                continue

    return False


def most_frequent(List):
    return max(set(List), key = List.count)