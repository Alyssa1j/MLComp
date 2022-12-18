import re
from matplotlib import test
import pandas as pd
import id3

def train_Income():
    features = {"age":0, "workclass":1, "fnlwgt":2, "education":3, "education-num":4, "marital-status":5, "occupation":6,
    "relationship":7, "race":8, "sex":9, "capital-gain":10, "capital-loss":11, "hours-per-week":12, "native-country":13}
    label_yes = [1]
    label_no = [0]

    income_data = pd.read_csv("MLComp/Decision_tree/data/train_final.csv", header=None)
   # print(len(income_data.index))
   # print(income_data.iloc[0])
    income_root = id3.ID3(income_data, features, label_yes, label_no, 14,100)
    #id3.printTree(income_root)
    test_data = pd.read_csv("MLComp/Decision_tree/data/test_final.csv")
    r,_ =test_data.shape

    result_df = pd.DataFrame(index=range(r),columns=range(2))
    result_df.columns = ["ID", "Prediction"]
    result_df["ID"] = test_data["ID"]
    id3.predict_data(income_root,test_data,features,result_df)
    result_df.to_csv('result_file.csv', index=False)

    return income_root

root=train_Income()
id3.printTree(root)
