import itertools
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_file_path = '/Users/ericsang/Desktop/Kaggle_Comp/Housing_Prices/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','OverallQual','OverallCond','GrLivArea','Fireplaces','YearRemodAdd','MSSubClass','ScreenPorch','WoodDeckSF','PoolArea']

def optfeats(features):
    X = home_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(predictions, val_y)

def optleafs(max_leaf_nodes, train_X, train_y, val_X, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=2)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(predictions, val_y)

X = home_data[features]
model = RandomForestRegressor(random_state=302)
model.fit(X,y)


test_data_path = '/Users/ericsang/Desktop/Kaggle_Comp/Housing_Prices/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','OverallQual','OverallCond','GrLivArea','Fireplaces','YearRemodAdd','MSSubClass','ScreenPorch','WoodDeckSF','PoolArea']]
test_preds = model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
