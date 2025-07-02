import pandas as pd
import numpy as np

data = pd.read_csv("data/housing.csv")
#print(data.head())

# the median income is a very important feature to predict the median housing price
# Creating a categorical feature based on the median income to sort the data into income categories
data["income_cat"] = pd.cut(data["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5])

# Creating a train-test split by stratified shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
	strat_train_set = data.loc[train_index]
	strat_test_set = data.loc[test_index]

# Now we can drop the income_cat column as it is no longer needed
for set_ in (strat_train_set, strat_test_set):
	set_.drop('income_cat', axis=1, inplace=True)

strat_train_set.to_csv('data/housing_strat_train_set.csv', index=False)
strat_test_set.to_csv('data/housing_strat_test_set.csv', index=False)

