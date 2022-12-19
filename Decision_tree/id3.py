
import numpy as np
import tree_node as tree
import id3_math as m


def ID3(data, attrs, label_yes, label_no,answer, depth=6):
    #print("ID3: ", depth)
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    weight = False
    for feature in attrs:
        gain = m.info_gain(data, attrs[feature], label_yes, answer)

        if gain > max_gain:
            if(feature == "fnlwgt"):
                weight = True
            else:
                max_gain = gain
                max_feat = feature
            #print(max_feat)
    
    if(max_gain == 0):
       # print(attrs)
        root.value = ("cluster grouping", answer)
        root.isLeaf = True
        root.pos = data[answer].value_counts()[1]
        root.neg =data[answer].value_counts()[0]
        root.pred = data[answer].value_counts().idxmax()
        
        pos = data[data[answer] == 1]
        neg = data[data[answer] == 0]
        fnlwgtPos = pos[attrs["fnlwgt"]]
        fnlwgtNeg = neg[attrs["fnlwgt"]]
        root.fnlwgtPos = fnlwgtPos.sum()
        root.fnlwgtNeg= fnlwgtNeg.sum()
        root.level=depth
        root.infogain = max_gain
        return root
#    if(weight):
 #       print("Max feature was fnlwgt")
    root.value = (max_feat, attrs[max_feat])
    root.level=depth
    root.infogain = gain
    if(type(data[attrs[max_feat]].iat[0]) is np.int64):
    #    print("int: ",examples[attr][0])
    #    print("Int Feature:",max_feat)
        threshold = np.median(data[attrs[max_feat]])
        subdata = data[data[attrs[max_feat]] <= threshold]
        subdata2 = data[data[attrs[max_feat]] > threshold]
        result1=-1.0
        result2=-1.0
        if(not subdata2.empty):
            result2 = m.entropy(subdata2, label_yes, answer)
        
        result1 = m.entropy(subdata, label_yes, answer)

        if  result1 == 0.0 or result2 == 0.0 or (depth-1)==1:
            if result1 == 0.0 or (depth-1)==1:
                root.children.append(generateLeaf(threshold,answer,result1,subdata,label_yes,label_no,depth))
            else:
                root.children.append(expandTree(threshold,subdata,attrs,max_feat,answer, label_yes, label_no, depth))
            if(not subdata2.empty):
                if result2 == 0.0 or (depth-1)==1:
                    root.children.append(generateLeaf(threshold+1,answer,result2,subdata2,label_yes,label_no,depth))
                else:
                    root.children.append(expandTree(threshold+1,subdata2,attrs,max_feat,answer, label_yes, label_no, depth))

        else:
            root.children.append(expandTree(threshold,subdata,attrs,max_feat,answer, label_yes, label_no, depth))
            if(not subdata2.empty):
                root.children.append(expandTree(threshold+1,subdata2,attrs,max_feat,answer, label_yes, label_no, depth))

    else:
        uniq = np.unique(data[attrs[max_feat]])
        for u in uniq:
            subdata = data[data[attrs[max_feat]] == u]
            result = m.entropy(subdata, label_yes, answer)
            #print("Result: ",u,"    ",result)
            if  result == 0.0 or (depth-1)==1:
                root.children.append(generateLeaf(u,answer,result,subdata,label_yes,label_no,depth))
            else:
                root.children.append(expandTree(u,subdata,attrs,max_feat,answer, label_yes, label_no, depth))

    return root

def generateLeaf(u,answer,result,subdata, label_yes, label_no, depth):
    newNode = tree.Node()
    newNode.isLeaf = True
    newNode.value = (u,answer)   

    pos = subdata[subdata[answer] == 1]
    neg = subdata[subdata[answer] == 0]
    fnlwgtPos = pos[2]
    fnlwgtNeg = neg[2]
    newNode.fnlwgtPos = fnlwgtPos.sum()
    newNode.fnlwgtNeg= fnlwgtNeg.sum()
    if((newNode.fnlwgtPos + newNode.fnlwgtNeg) != 0):
           # print("-->Percentage Pred: ", root.pos/(root.pos + root.neg))
            newNode.pred= newNode.fnlwgtPos/(newNode.fnlwgtPos + newNode.fnlwgtNeg)
    newNode.entropy = result
    newNode.level = depth-1  
    return newNode

def expandTree(u,subdata,attrs,max_feat,answer, label_yes, label_no, depth):
    dummyNode = tree.Node()
    dummyNode.value = (u, attrs[max_feat])
    #print(dummyNode.value)
    #calculate current prediction value
    pos = subdata[subdata[answer] == 1]
    neg = subdata[subdata[answer] == 0]
    fnlwgtPos = pos[2]
    fnlwgtNeg = neg[2]
    dummyNode.fnlwgtPos = fnlwgtPos.sum()
    dummyNode.fnlwgtNeg= fnlwgtNeg.sum()
    if((dummyNode.fnlwgtPos + dummyNode.fnlwgtNeg) != 0):
           # print("-->Percentage Pred: ", root.pos/(root.pos + root.neg))
            dummyNode.pred= dummyNode.fnlwgtPos/(dummyNode.fnlwgtPos + dummyNode.fnlwgtNeg)
    
    dummyNode.level=depth-1
    new_attrs = attrs.copy()
    new_attrs.pop(max_feat)      
    child = ID3(subdata, new_attrs, label_yes,label_no, answer,depth-2)
    dummyNode.children.append(child)
    return dummyNode

def printTree(root, depth=0):
        for i in range(depth):
            print("\t", end="")
        if(root is None):
            return
        print(root.value,root.level, end="")
        if root.isLeaf:
            if((root.fnlwgtPos + root.fnlwgtNeg) != 0):
                auc=root.fnlwgtPos/(root.fnlwgtPos + root.fnlwgtNeg)
                print(" -> ", root.pred, "Percentage: ", auc)
            else:
                print(" -> ", root.pred)
        print()
        #if(not root.children):
         #   print("Tree Node: ", root.value," my answer is: ", root.pred, " leaf? ", root.isLeaf)
        for child in root.children:
            printTree(child, depth + 1)

def prediction_r(root, dataset, features,label_yes,label):
    rows,colms = dataset.shape
    correct =0
    for _,data_row in dataset.iterrows():
      #  print("data_row:",data_row)
        b = prediction(root, data_row, features, label_yes,label)
        if b:
            correct+=1
    return correct/rows

def predict_data(root,dataset,features,rdf):
    for i,data_row in dataset.iterrows():
        #print("data_row:",data_row[1:14])
        result = prediction_result(i, root, data_row[1:15], features)
        rdf["Prediction"].iat[i] = result
    return

def prediction_result(i, root, data_row, features):
    if root.isLeaf:
        if((root.fnlwgtPos + root.fnlwgtNeg) != 0):
            return root.fnlwgtPos/(root.fnlwgtPos + root.fnlwgtNeg)
        else:
            return root.pred
    else:
        num = False
        rootPred = (root.fnlwgtPos/(root.fnlwgtPos + root.fnlwgtNeg) if ((root.fnlwgtPos + root.fnlwgtNeg) != 0) else 0.0)
        for c in root.children:
            if(type(data_row[features[root.value[0]]]) is int):
                if(num):
                    if(c.isLeaf):
                        return prediction_result(i,c, data_row, features) + (rootPred/(101-c.level))
                    else:
                        return prediction_result(i,c.children[0], data_row, features)+ (rootPred/(101-c.level))
                if(c.value[0] <= data_row[features[root.value[0]]]):
                    if(c.isLeaf):
                        return prediction_result(i,c, data_row, features)+ (rootPred/(101-c.level))
                    else:
                        return prediction_result(i,c.children[0], data_row, features)+ (rootPred/(101-c.level))
                else:
                    num = True
            if(c.value[0] == data_row[features[root.value[0]]]):
                if(c.isLeaf):
                    return prediction_result(i,c, data_row, features)+ (rootPred/(101-c.level))
                else:
                    return prediction_result(i,c.children[0], data_row, features)+ (rootPred/(101-c.level))
            else:
                continue
        
            #branch all and average result
        total = len(root.children)
        val =0
        for c in root.children:
            if(c.isLeaf):
                val += prediction_result(i,c, data_row, features)+ (rootPred/(101-c.level))
            else:
                val += prediction_result(i,c.children[0], data_row, features)+ (rootPred/(101-c.level))
        return val/total


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