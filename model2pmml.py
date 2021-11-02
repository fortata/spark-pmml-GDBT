
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn.feature_selection import SelectKBest

from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, LabelEncoder,OneHotEncoder, MinMaxScaler,StandardScaler,Normalizer
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer


def all_classifiers_test():
    data=load_iris()
    x=data.data
    y=data.target
    df_x = pd.DataFrame(x)
    df_x.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    return df_x,y

    GBDT = GradientBoostingClassifier()
    mapper = DataFrameMapper([ # data process
              (["Sepal.Length"],FunctionTransformer(np.abs)),
              (["Sepal.Width"],[MinMaxScaler(), SimpleImputer()]),
              (["Petal.Length"],None),
              (["Petal.Width"],OneHotEncoder()),
            (['Petal.Length', 'Petal.Width'],
             [MinMaxScaler(),StandardScaler()])
    ])

    iris_pipeline = PMMLPipeline([
        ("mapper", mapper),
        ("pca", PCA(n_components=3)),
        ("selector", SelectKBest(k=2)), 
        ("classifier", GBDT)])
    iris_pipeline.fit(df_x, y)
    # iris_pipeline.fit(X_train.values, y_train)
    sklearn2pmml(iris_pipeline, "GBDT.pmml", with_repr=True)
if __name__=="__main__":
      all_classifiers_test()
