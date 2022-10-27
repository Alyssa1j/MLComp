import numpy as np
import id3

# Decision stump used as weak classifier
class DecisionStump:
    def __init__(self):
        self.tree = None
        self.feature_idx = None
        self.alpha = None

    def predict(self, data,features, pos_label,ly):
       # id3.printTree(self.tree)
        n_samples = data.shape[0]
        predictions = np.ones(n_samples)
        for i,data_row in data.iterrows():
            b = id3.prediction(self.tree, data_row, features, pos_label,ly)
            if b:
              #  print("correct")
                predictions[i] = -1
            else:
               # print("false")
                predictions[i] = 1
                
        return predictions