# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings
'''
To install any library from those:
  pip install ["the name of your library"]
  example:
  pip install pandas

'''
## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
## for machine learning
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from lime import lime_tabular

dtf = pd.read_csv("train.csv")
cols = ["MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities"]
dtf =  dtf[["Id"] + cols+ ["SalePrice"]]
dtf.head(10)
'''
Recognize whether a column is numerical or categorical.
:parameter
    :param dtf: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
:return
    "cat" if the column is categorical or "num" otherwise
'''
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"

dic_cols = { col:utils_recognize_type(dtf, col, max_cat=20) for col in dtf.columns}
print (dic_cols)
heatmap = dtf.isnull()
for k,v in dic_cols.items():
    if v== "num":
        heatmap[k]= heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    else:
        heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
plt.show()
print("\033[1;37;40m Categerocoal ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN")
dtf =  dtf.set_index("Id")
dtf = dtf.rename(columns={"SalePrice": "Y"})
##split data
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
##print info
print ("X_train shape:", dtf_train.drop("Y", axis=1).shape, "| X_test shape",
       dtf_test.drop("Y", axis=1).shape)
print ("y_train mean: ", round (np.mean(dtf_train["Y"]),2) , "| y_test mean:", round(np.mean(dtf_test["Y"]),2))
print(dtf_train.shape[1], "features:", dtf_train.drop("Y",axis=1).columns.to_list())
dtf_train["LotFrontage"] =  dtf_train["LotFrontage"].fillna(dtf_train["LotFrontage"].mean())
dtf_train =dtf_train .drop(columns="Alley")
##print(dtf_train.shape[1], "features:", dtf_train.drop("Y",axis=1).columns.to_list())
##Encoding one_hot
dummy = pd.get_dummies(dtf_train["MSSubClass"],
                       prefix="MSSubClass_cluster",drop_first=True)
dtf_train= pd.concat([dtf_train, dummy], axis=1)
print( dtf_train.filter(like="MSSubClass_cluster",axis=1).head() )
dtf_train = dtf_train.drop("MSSubClass", axis=1)
dummy = pd.get_dummies(dtf_test["MSSubClass"],
                       prefix="MSSubClass_cluster",drop_first=True)
dtf_test= pd.concat([dtf_test, dummy], axis=1)
print( dtf_test.filter(like="MSSubClass_cluster",axis=1).head() )
dtf_test = dtf_test.drop("MSSubClass", axis=1)
corr_matrix = dtf_test.corr(method="pearson")
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")
X_names = dtf_train.reindex(columns=['LotFrontage', 'LotArea', 'MSSubClass_cluster_90'])
X_names=X_names.columns.to_list()
print(X_names)
X_train = dtf_train[X_names].values
y_train = dtf_train["Y"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["Y"].values

## call model
model = linear_model.LinearRegression()
## K fold validation
scores = []
cv = model_selection.KFold(n_splits=5, shuffle=True)
fig = plt.figure()
i = 1
for train, test in cv.split(X_train, y_train):
    prediction = model.fit(X_train[train],
                 y_train[train]).predict(X_train[test])
    true = y_train[test]
    score = metrics.r2_score(true, prediction)
    scores.append(score)
    plt.scatter(prediction, true, lw=2, alpha=0.3,
                label='Fold %d (R2 = %0.2f)' % (i,score))
    i = i+1
plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)],
         linestyle='--', lw=2, color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('K-Fold Validation')
plt.legend()
plt.show()