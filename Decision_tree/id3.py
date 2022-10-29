
import numpy as np
import tree_node as tree
import id3_math as m


def ID3(data, attrs, label_yes, label_no,answer, depth=6):
    #print("ID3: ", depth)
    root = tree.Node()

    max_gain = 0
    max_feat = ""
    for feature in attrs:
        gain = m.info_gain(data, attrs[feature], label_yes, answer)

        if gain > max_gain:
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
        root.level=depth
        root.infogain = max_gain
        return root
    root.value = (max_feat, attrs[max_feat])
    root.level=depth
    root.infogain = gain
    if(type(data[attrs[max_feat]].iat[0]) is np.int64):
    #    print("int: ",examples[attr][0])
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
    newNode.pred = np.unique(subdata[answer])
  #  print("Before check ",newNode.value, newNode.pred)
  #  print(subdata)
    if(any(p in newNode.pred for p in label_no) and any(p in newNode.pred for p in label_yes)):
        newNode.pred = subdata[answer].value_counts().idxmax()
   #     print("after check: ",newNode.value, newNode.pred)
    newNode.entropy = result
    newNode.level = depth-1  
    return newNode

def expandTree(u,subdata,attrs,max_feat,answer, label_yes, label_no, depth):
    dummyNode = tree.Node()
    dummyNode.value = (u, attrs[max_feat])
    #print(dummyNode.value)
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
            if((root.pos + root.neg) != 0):
                auc=root.pos/(root.pos + root.neg)
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
        if((root.pos + root.neg) != 0):
            return root.pos/(root.pos + root.neg)
                
        else:
            return root.pred
    else:
        for c in root.children:
            #compare values then pick the path that fits.
            #check for integer types
        #    print("values: ",c.value[0], root.value[0])
         #   print(len(data_row))
          #  print(data_row)
            #print(i, " -->", data_row[features[root.value[0]]])
            if(type(data_row[features[root.value[0]]]) is np.int64):
               # print("I am an integer")
                if(c.value[0] <= data_row[features[root.value[0]]]):
                    if(c.isLeaf):
                        return prediction_result(i,c, data_row, features)
                    else:
                        return prediction_result(i,c.children[0], data_row, features)

            if(c.value[0] == data_row[features[root.value[0]]]):
                if(c.isLeaf):
                    return prediction_result(i,c, data_row, features)
                else:
                    return prediction_result(i,c.children[0], data_row, features)
            else:
                continue

    return 0

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