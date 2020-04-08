from tree import DecisionTreeRegressor
import tree
import numpy as np


def batch_generator(X, y, batch_size):
    perm = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idxs = perm[i: i + batch_size]
        yield X[idxs], y[idxs]

class DecisionForest(DecisionTreeRegressor):
    def __init__(self, x, y, max_depth, number=1):
        super().__init__(x, y, max_depth)
        self.number = number
        self.root = self.fit1(self.x, self.y)
        
    def fit1(self, x, y):
        forest_fit = []
        batch_size = int(len(x)/self.number)
        for x_batch, y_batch in batch_generator(x, y, batch_size):
            forest_fit.append(self.fit(x_batch, y_batch))
        return forest_fit 
 
    def predict1(self, x):
        result = []
        for j in range(self.number):
            y = np.zeros(x.shape[0])
            for i, row in enumerate(x):
                node = self.root[j]
                while not isinstance(node, tree.Leaf):
                    if row[node.feature] > node.value:
                        node = node.right
                    else:
                        node = node.left
                y[i] = node.average
            result.append(y)
        return np.mean(np.array(result), axis=0)
