# Library imports
import timeit
import findspark
findspark.init()

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

import math
import random
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer

# Starting time, for execution time calculation.
start = timeit.default_timer()

# Initializing Spark session
sc = SparkContext('local')
spark = SparkSession(sc)

# Reading the train-test dataframe from a CSV file.
train_sp = spark.read.csv(r"..\data\train_test_df.csv", header=True, inferSchema=True)
print(train_sp.dtypes)

train_sp.show()

# Setting the number of replications to run
n_replications = 20000

# Replicating the spark dataframe into multiple copies
replication_df = spark.createDataFrame(pd.DataFrame(list(range(1, n_replications + 1)), columns=['run_id']))

replicated_train_df = train_sp.crossJoin(replication_df)
replicated_train_df.show()

# Declaring the schema for the output spark dataframe
outSchema = StructType([StructField('run_id', IntegerType(), True),
                        StructField('avg_R2', DoubleType(), True),
                        StructField('avg_MAE', DoubleType(), True),
                        StructField('avg_RMSE', DoubleType(), True),
                        StructField('n_estimators', IntegerType(), True),
                        StructField('max_depth', IntegerType(), True),
                        StructField('min_child_weight', IntegerType(), True),
                        StructField('eta', DoubleType(), True),
                        StructField('subsample', DoubleType(), True),
                        StructField('colsample_bytree', DoubleType(), True),
                        StructField('gamma', DoubleType(), True),
                        StructField('alpha', DoubleType(), True),
                        StructField('lambda', DoubleType(), True),
                        StructField('n_neighbors', IntegerType(), True),],)


# Decorating the function with pandas_udf decorator
@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def run_model(pdf):
    run_id = pdf.run_id.values[0]

    # Setting model hyperparameter values
    n_estimators = random.choice(list(range(50, 500)))
    max_depth = random.choice(list(range(2, 30)))
    min_child_weight = random.choice(list(range(1, 40)))
    eta = random.choice(list(np.linspace(0.001, 0.1, 100)))
    eta = round(eta, 6)
    subsample = random.choice(list(np.linspace(0.1, 1, 100)))
    subsample = round(subsample, 3)
    colsample_bytree = random.choice(list(np.linspace(0.1, 1, 100)))
    colsample_bytree = round(colsample_bytree, 3)
    gamma = random.choice(list(np.linspace(0, 10, 100)))
    gamma = round(gamma, 3)
    reg_alpha = random.choice(list(np.linspace(0, 10, 100)))
    reg_alpha = round(reg_alpha, 3)
    reg_lambda = random.choice(list(np.linspace(1, 10, 100)))
    reg_lambda = round(reg_lambda, 3)

    # Selecting feature columns
    feature_columns = ['WELL', 'LAT', 'LON', 'DEPT', 'CALI', 'DT', 'RESD', 'NPHI', 'RHOB', 'GR', 'CLUSTER']

    # Separating feature and target variables
    X = pdf[feature_columns]
    y = pdf['TOC']

    # Defining the sample distance for data imputation
    n_neighbors = random.choice(list(range(10, 1000)))

    # Performing data imputation
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')

    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Configuring the model
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, eta=eta, subsample=subsample,
                         colsample_bytree=colsample_bytree, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda,verbosity=2)

    # Train-test stratified k-fold cross-validation
    n_splits = 10
    num_bins = math.floor(len(pdf) / n_splits)
    bins_on = pdf['TOC']
    qc = pd.cut(bins_on.tolist(), num_bins)
    pdf['target_bins'] = qc.codes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for train_index, test_index in skf.split(X, pdf['target_bins']):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        test_data_prediction = model.predict(X_test)

        # Obtaining R2, RMSE and MAE scores for each fold
        r2_score = metrics.r2_score(y_test, test_data_prediction)
        rmse_score = metrics.mean_squared_error(y_test, test_data_prediction, squared=True)
        mae_score = metrics.mean_absolute_error(y_test, test_data_prediction)

        r2_scores.append(r2_score)
        rmse_scores.append(rmse_score)
        mae_scores.append(mae_score)

    avg_r2 = sum(r2_scores) / len(r2_scores)
    avg_rmse = sum(rmse_scores) / len(rmse_scores)
    avg_mae = sum(mae_scores) / len(mae_scores)

    # Adding parameter`s values and scores to a summary dataframe
    res = pd.DataFrame({'run_id': run_id, 'avg_R2': avg_r2, 'avg_MAE': avg_mae, 'avg_RMSE': avg_rmse,
                        'n_estimators': n_estimators, 'max_depth': max_depth, 'min_child_weight':min_child_weight, 'eta': eta,
                        'subsample': subsample, 'colsample_bytree': colsample_bytree,
                        'gamma': gamma, 'alpha': reg_alpha, 'lambda': reg_lambda,'n_neighbors': n_neighbors}, index=[0])

    return res


# Exhibiting the summary of the top 20 runs
results = replicated_train_df.groupby("run_id").apply(run_model)
print(results.sort(F.desc("avg_RMSE")).show(20))

# End time, for execution time calculation
end = timeit.default_timer()
print(f'Total time of execution was {end-start} s')

