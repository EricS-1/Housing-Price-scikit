import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from itertools import combinations

iowa_file_path = '/Users/ericsang/Desktop/Kaggle_Comp/Housing_Prices/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# optomized features
features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','OverallQual','OverallCond','GrLivArea','Fireplaces','YearRemodAdd','MSSubClass','ScreenPorch','WoodDeckSF','PoolArea']

def findfeatmae(features):
    '''
    Finds the mean absolute error of a model given a set of features

    Parameters
    ----------
    features : list
        List of str that represent every feature related to House Pricing
        
    Returns
    -------
    float
        the mean absolute error of the model
    '''
    X = home_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(predictions, val_y)

def optfeats(features):
    '''
    Finds the lowest MAE out of every combination of each feature

    Parameters
    ----------
    features : list
        List of str that represent every feature related to House Pricing
        
    Returns
    -------
    list
        a list of strings representing the optomized list
    '''
    allmae = []
    allcomb =[]

    for i in range(1,len(features)+1):
        for j in list(combinations(features, i)):
            allcomb.append(j)
            allmae.append(findfeatmae(j))

    return allcomb[allmae.index(min(allmae))]
        
def findleafmae(max_leaf_nodes, train_X, train_y, val_X, val_y):
    '''
    finds the mae of a model with a set number of leaf nodes

    Parameters
    ----------
    max_leaf_nodes : int
        number of leaf nodes
    train_X : list
        the training data for fitting
    train_y : list
        the target values of the training data for fitting
    val_X : list
        the validation data to find mae
    val_y : list
        the validation data to find mae
        
    Returns
    -------
    float
        the mean absolute error of the model
    '''
    
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=2)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(predictions, val_y)

X = home_data[features]
model = RandomForestRegressor(random_state=302)
model.fit(X,y)

test_data_path = '/Users/ericsang/Desktop/Kaggle_Comp/Housing_Prices/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
