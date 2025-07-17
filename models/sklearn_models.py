from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class RandomForest:
    def __init__(self, n_estimators=100, criterion="squared_error"):
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)

    def fit(self, x, y):
        self.model.fit(x, y.reshape(-1))
    
    def predict(self, x):
        return self.model.predict(x)[:, None]


class SupportVector:
    def __init__(self):
        self.model = SVR(kernel="rbf")

    def fit(self, x, y):
        self.model.fit(x, y.reshape(-1))
    
    def predict(self, x):
        return self.model.predict(x)[:, None]


class KNeighbors:
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

    def fit(self, x, y):
        self.model.fit(x, y.reshape(-1))
    
    def predict(self, x):
        return self.model.predict(x)[:, None]