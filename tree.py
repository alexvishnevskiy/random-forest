import numpy as np
from collections import namedtuple

Leaf = namedtuple('Leaf', ('average', 'x'))
Node = namedtuple('Node', ('feature', 'value', 'left', 'right'))
Split = namedtuple('Split', ('mse', 'value'))


class DecisionTreeRegressor:
    def __init__(self, x, y, max_depth=np.inf):
        self.x = np.atleast_2d(x)
        self.y = np.atleast_1d(y)
        self.features = x.shape[1]
        self.max_depth = max_depth
        self.root = self.fit(self.x, self.y)

    def mse(self, y):
        return np.sum((np.mean(y) - y) ** 2)/np.sum(y)

    def fit(self, x, y, depth=1):
        x = np.array(x)
        y = np.array(y)
        if depth > self.max_depth or self.mse(y) < 1e-6:
            return Leaf(np.mean(y), x)  # должен вернуть среднее выборки
        feature = self.best_feature(x, y)
        value = self.split(x[:, feature], y).value
        left_xy, right_xy = self.partition(x, y, value, feature)
        left = self.fit(*left_xy, depth=depth + 1)
        right = self.fit(*right_xy, depth=depth + 1)
        return Node(feature, value, left, right)

    def partition(self, x, y, value, feature):
        i = x[:, feature] > value 
        j = np.logical_not(i)
        return (x[j], y[j]), (x[i], y[i])

    def split(self, x, y):
        t = np.array(sorted(list(zip(x, y))))
        z = [((len(t[:i, 1])*self.mse(t[:i, 1]) + len(t[i:, 1])*self.mse(t[i:, 1])))/(len(t)) for i in range(1, len(t))]
        razbienie = z.index(min(z))
        return Split(min(z), t[razbienie, 0])

    def best_feature(self, x, y):
        feature = [self.split(x[:, i], y).mse for i in range(self.features)]
        best_feature = feature.index(min(feature))
        return best_feature

    def predict(self, x):
        y = np.zeros(x.shape[0])
        for i, row in enumerate(x):
            node = self.root
            while not isinstance(node, Leaf):
                if row[node.feature] > node.value:
                    node = node.right
                else:
                    node = node.left
            y[i] = node.average
        return y