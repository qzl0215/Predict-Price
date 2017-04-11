from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd

X = pd.read_csv('boscun_X.csv').values
y = pd.read_csv('boscun_y.csv').values
# random selection
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(X, y, train_size=0.8,
                                                                                        random_state=1)
# MLP Regression Model
#model_mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(a, b), random_state=1)
for number_of_estimator in [200]:
    model_random_forest = RandomForestRegressor(n_estimators= number_of_estimator)
    model_random_forest.fit(train_x_disorder, train_y_disorder.ravel())
    mlp_score = model_random_forest.score(test_x_disorder, test_y_disorder.ravel())
    print('Score of regression model is ', mlp_score)

#lr是一个LogisticRegression模型
joblib.dump(model_random_forest, 'model_mlp.model')
#model_mlp = joblib.load('model_mlp.model')