from numpy import logspace
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def best_train_partition_search(smodel, x, y, p=[-2,-1,10]):
    test_size_grid = 6*logspace(p[0],p[1],p[2])
    test_score_grid = []
    for i in test_size_grid:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i, random_state=102)
        smodel.fit(X_train,y_train)
        test_score_grid.append(accuracy_score(smodel.predict(X_test), y_test))
    return test_size_grid, test_score_grid