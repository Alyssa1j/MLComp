import pandas as pd
import id3

#data columns are
#0      1       2       3       4           5       6
#buying, maint, doors, persons, lug_boot, safety, label
buying= ["low", "med", "high", "vhigh"]
maint = ["low", "med", "high", "vhigh"]
doors = ["2", "3", "4", "5more"]
person = ["2", "4", "more"]
lug_boot = ["small", "med", "big"]
safety= ["low", "med", "high"]

features = {"buying":0, "maint":1, "doors":2, "person":3, "lug_boot":4, "safety":5}
label_yes = ["acc", "good", "vgood"]
label_no = ["unacc"]

def car_train():
    #changing the label to be binary 
    car_data = pd.read_csv("Decision_Tree/data/car/train.csv", header=None)
    root = id3.ID3(car_data, features, label_yes, 6)
    return root

def car_test():
    test_data = pd.read_csv("data/car/test.csv", header=None)
    test_root = id3.ID3(test_data, features, label_yes, 6)
    return test_root

def test_Tennis(method):
    tennis_data = pd.read_csv("Decision_Tree/data/playtennis.csv", header=None)
    
    feat={"Outlook":0, "Temperature":1, "Humidity":2, "Wind":3}
   
    l_y = ["yes"]
    l_n = ["no"]
    tennis_root = id3.ID3(tennis_data, feat, l_y, l_n, 4,2)
    return tennis_root

ly = ["yes"]
label = "answer"
feat={"Outlook":0, "Temperature":1, "Humidity":2, "Wind":3, "answer":4}
play_tennis = test_Tennis("IG")
id3.printTree(play_tennis)
test_data_Tennis = pd.read_csv("Decision_Tree/data/test_Tennis.csv", header=None)
result1 = id3.prediction_result(play_tennis, test_data_Tennis,feat, ly, 4)
print(result1)

#test_data_car = pd.read_csv("Decision_Tree/data/car/test.csv", header=None)
#train_car = car_train()
#result2 = id3.prediction_result(train_car, test_data_car, features, label_yes, 6)
#print(result2)
