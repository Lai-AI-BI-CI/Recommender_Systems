# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 08:10:37 2022

@author: lai.yeung
"""

import os
import logging
import time
# import python-snappy
# import pyarrow
# conda install python-snappy
import pandas as pd
# import datatable as dt
from datetime import date, timedelta
import numpy as np
import re
# import PySpark
# os.chdir("D:/pyspark_directory/A0025")
# from udf_config import timer, reduce_memory_usage
# import ROEM_cv
import gc

import findspark
findspark.init()
findspark.find()
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
# from pyspark.sql import DataFrame
from recommenders.utils.timer import Timer


print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("PySpark version: {}".format(pyspark.__version__))

# spark.SQLContext.clearCache()

# spark.sql.Catalog.clearCache()

# # In order to remove all cached tables we can use :
# spark.sqlContext.clearCache()

# # Un-cache all tables in the Spark session:
# spark.catalog.clearCache()

# df.unpersist()

################## pyarrow import method
# import pyarrow.parquet as pq
# raw_2021_order=pq.read_table("D:/pyspark_directory/A0025_result_raw_(2021H2)_2022-02-21.parquet")
# raw_2021_order=raw_2021_order.to_pandas()

# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

os.environ["SPARK_LOCAL_DIRS"] = "D:/Users/Lai.Yeung.2/Temp"

# .config("spark.executor.processTreeMetrics", "false").getOrCreate()
spark = SparkSession.builder.master("local[*]").appName("SparkByExamples.com").enableHiveSupport()\
    .config('spark.sql.execution.arrow.pyspark.enabled', True) \
    .config('spark.sql.session.timeZone', 'UTC') \
    .config('spark.driver.memory','24G') \
    .config('spark.driver.maxResultSize', "4g") \
    .config('spark.sql.shuffle.partitions',1200) \
    .config('spark.default.parallelism',1200) \
    .config('spark.sql.repl.eagerEval.enabled', True) \
    .config('spark.worker.cleanup.enabled', True) \
    .getOrCreate()
    # .config('spark.ui.showConsoleProgress', True) \
    # .config('spark.worker.cleanup.interval', 900) \
    # .config('spark.worker.cleanup.appDataTtl', 86400) \

spark.sparkContext.setLogLevel('WARN')

# sc = spark.sparkContext

################## Spark Test Installation
# rdd=spark.sparkContext.parallelize([1, 2, 3, 4, 56])
# print(rdd.count())

parqDF = spark.read.parquet("D:/Users/Lai.Yeung.2/A0025/A0025_result_raw_order_(2021).parquet")

# parqDF.printSchema()

# Add Columns
parqDF=parqDF.withColumn('PGG_PROD_GRP', parqDF['PROD_GRP'])
parqDF=parqDF.withColumn('PGG', split(parqDF['PGG_PROD_GRP'], ':').getItem(0))
parqDF=parqDF.withColumn('PROD_GRP', split(parqDF['PGG_PROD_GRP'], ':').getItem(1))

parqDF = parqDF.filter((parqDF["SITE"] == "YSGB")\
                       & (~parqDF["BRAND"].isin(["YesStyle Gift", "YesStyle Sample"]))\
                       & (parqDF["SHIP_TO_COUNTRY"] == "United States")\
                       & (parqDF["PROD_REV_USD"] > 0))
    

# =============================================================================
# # parqDF.select(df['name'], df['age'] + 1).show()
# # df.filter(df['age'] > 21).show()
# # df.groupBy("age").count().show()
# =============================================================================

data=parqDF.select("CUST_ID", "ORDER_DT", "PGG_PROD_GRP"\
                   ,when(parqDF["PROD_TITLE_PARENT_PROD_ID"].isNull(), parqDF["PROD_ID"]).otherwise(parqDF["PROD_TITLE_PARENT_PROD_ID"]).alias("PROD_TITLE_PARENT_PROD_ID")\
                   ,when(parqDF["PROD_TITLE_PARENT_PROD_ID"].isNull(), parqDF["PROD_TITLE"]).otherwise(parqDF["PROD_TITLE_PARENT_TITLE"]).alias("PROD_TITLE_PARENT_TITLE")\
                   , "PROD_ID", "PROD_TITLE", "PROD_QTY", "PROD_REV_USD")


# check_data_des = data.toPandas().describe()

from pyspark.sql.functions import monotonically_increasing_id
CUST_ID_table = data.select("CUST_ID").distinct().coalesce(1)
CUST_ID_table = CUST_ID_table.withColumn("CUST_ID_Int", monotonically_increasing_id()).persist()

PARENT_PROD_ID_table = data.select("PROD_TITLE_PARENT_PROD_ID").distinct().coalesce(1)

# Add zeros
cross_join = CUST_ID_table.crossJoin(PARENT_PROD_ID_table).join(data, ["CUST_ID", "PROD_TITLE_PARENT_PROD_ID"], "left").fillna(0)

# cross_join.printSchema()

# PARENT_PROD_ID_table.filter("PROD_TITLE_PARENT_PROD_ID is NULL").show(5)

# =============================================================================
# # record_table=record_table.join(CUST_ID_table, record_table["CUST_ID"] == CUST_ID_table["CUST_ID"], "inner").select(record_table["*"], CUST_ID_table["CUST_ID_num"]).cache()
# 
# # CUST_ID_table2 = spark.sql("SELECT CUST_ID, row_number() over (PARTITION BY CUST_ID ORDER BY CUST_ID) as CUST_ID_num from (select distinct(CUST_ID) CUST_ID from temp)")
# =============================================================================

# cross_join = cross_join.select("CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID", "Purchases")

# (training, validating, testing) = cross_join.drop("CUST_ID").randomSplit([0.6, 0.2, 0.2], seed = 0)

# training=training.cache()
# validating=training.cache()

# Sparsity
def Sparsity(ratings_df, user, item):
    numerator = ratings_df.count()
    num_users = ratings_df.select(user).distinct().count()
    num_items = ratings_df.select(item).distinct().count()
    
    denominator = num_users*num_items
    
    sparsity = (1.0-(numerator*1.0)/denominator)*100
    
    # print ("Sparsity: "+str(sparsity))
    return {"num_users": f"{num_users:,}", "num_items": f"{num_items:,}", "numerator": f"{numerator:,}", "denominator": f"{denominator:,}", "empty_percent": f"{sparsity:.2f}%"}

sparsity=Sparsity(ratings_df = data, user = "CUST_ID", item = "PROD_TITLE_PARENT_PROD_ID")


# Add CUST_ID_Int
data = data.join(CUST_ID_table, ["CUST_ID"], "left")

# a=pd.DataFrame(data.head(10), columns=data.columns) 

# a=pd.DataFrame(CUST_ID_table.head(10), columns=CUST_ID_table.columns) 

# TOP_K=50
COL_USER="CUST_ID"
COL_ITEM="PROD_TITLE_PARENT_PROD_ID"
# COL_RATING="PROD_QTY"
# COL_PREDICTION="prediction"
COL_TIMESTAMP = "ORDER_DT"

data = data.toPandas()

from recommenders.datasets.python_splitters import (
    python_random_split, 
    python_chrono_split, 
    python_stratified_split
)

data_train, data_test = python_chrono_split(
    data, ratio=0.6, filter_by="user",
    col_user=COL_USER, col_item=COL_ITEM, col_timestamp=COL_TIMESTAMP
)

data_train.shape[0]+data_test.shape[0]

data_valid, data_test = python_chrono_split(
    data_test, ratio=0.5, filter_by="user",
    col_user=COL_USER, col_item=COL_ITEM, col_timestamp=COL_TIMESTAMP
)

# data_train.shape[0]+data_valid.shape[0]+data_test.shape[0]

# data_train[data_train[COL_USER] == 1].tail(10)

# data_test[data_test[COL_USER] == 1].head(10)


data_train = spark.createDataFrame(data_train)
data_valid = spark.createDataFrame(data_valid)
data_test = spark.createDataFrame(data_test)

data_train.write.parquet("D:/Users/Lai.Yeung.2/A0025/data_train.parquet")
data_valid.write.parquet("D:/Users/Lai.Yeung.2/A0025/data_valid.parquet")
data_test.write.parquet("D:/Users/Lai.Yeung.2/A0025/data_test.parquet")

data_train = spark.read.parquet("D:/Users/Lai.Yeung.2/A0025/data_train.parquet")
data_valid = spark.read.parquet("D:/Users/Lai.Yeung.2/A0025/data_valid.parquet")
data_test = spark.read.parquet("D:/Users/Lai.Yeung.2/A0025/data_test.parquet")


# a=pd.DataFrame(data_valid.filter("CUST_ID == 'ff80808125967bdc01259ffd05d60090'").tail(10), columns=data_valid.columns)
# a2=pd.DataFrame(data_test.filter("CUST_ID == 'ff80808125967bdc01259ffd05d60090'").head(10), columns=data_test.columns) 
# Calculate the weighted count with time decay.

# T = 180
# t_ref = pd.to_datetime(data2_w['Timestamp']).max()
# 
# data2_w['Timedecay'] = data2_w.apply(
#     lambda x: x['Weight'] * np.power(0.5, (t_ref - pd.to_datetime(x['Timestamp'])).days / T), 
#     axis=1
# )

T = 180
t_ref = data_train.select(max(COL_TIMESTAMP)).collect()[0][0]

data_train = data_train.withColumn("Timedecay", col("PROD_QTY")*pow(0.5, datediff(lit(t_ref), col(COL_TIMESTAMP))/lit(T)))

# Register the DataFrame as a SQL temporary view
data_train.createOrReplaceTempView("temp")

# Beauty Products
# PROD_TITLE_PARENT_PROD_ID : fill na 
# Raw Data Problem: e.g. PROD_TITLE_PARENT_PROD_ID=1095172519 have two different PGG_PROD_GRP ("BD Brand:YS Beauty B", "BD Brand:YS HKFashion F01")
data_train_final = spark.sql("SELECT CUST_ID_Int, CUST_ID, PROD_TITLE_PARENT_PROD_ID, sum(PROD_QTY) as PROD_QTY_SUM, max(ORDER_DT) as PARENT_PROD_LAST_ORDER_DT\
                         , sum(Timedecay) as Timedecay_SUM\
                         from temp\
                          where PGG_PROD_GRP like '%Beauty%'\
                         group by CUST_ID_Int, CUST_ID, PROD_TITLE_PARENT_PROD_ID \
                         having PROD_QTY_SUM > 2")

# a=pd.DataFrame(record_table.head(10), columns=record_table.columns)    

sparsity=Sparsity(ratings_df = data_train_final, user = "CUST_ID", item = "PROD_TITLE_PARENT_PROD_ID")

# num_users:59164
# num_users:6139
# denominator:363,207,796
# cv_time:4368 seconds

data_train_final = data_train_final.select("CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID", "Timedecay_SUM")
data_valid_final = data_valid.where("PGG_PROD_GRP like '%Beauty%'").select("CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID", col("PROD_QTY").alias("Timedecay_SUM"))
data_test_final = data_test.where("PGG_PROD_GRP like '%Beauty%'").select("CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID", col("PROD_QTY").alias("Timedecay_SUM"))

#Building 5 folds within the training set.
train1, train2, train3, train4, train5 = data_train_final.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed = 1)
fold1 = train2.union(train3).union(train4).union(train5)
fold2 = train3.union(train4).union(train5).union(train1)
fold3 = train4.union(train5).union(train1).union(train2)
fold4 = train5.union(train1).union(train2).union(train3)
fold5 = train1.union(train2).union(train3).union(train4)

foldlist = [(fold1, train1), (fold2, train2), (fold3, train3), (fold4, train4), (fold5, train5)]

# =============================================================================
# model start
# =============================================================================
from pyspark.ml.recommendation import ALS

TOP_K=20
COL_USER="CUST_ID_Int"
COL_ITEM="PROD_TITLE_PARENT_PROD_ID"
COL_RATING="Timedecay_SUM"
COL_PREDICTION="prediction"
# COL_TIMESTAMP = "Timestamp"

COL_DICT = {
    "col_user": COL_USER,
    "col_item": COL_ITEM,
    "col_rating": COL_RATING,
    "col_prediction": COL_PREDICTION,
}

from recommenders.tuning.parameter_sweep import generate_param_grid
# Hyperparameter Search
param_dict = {    
    "rank": [10],
    "maxIter": [10],
    "regParam": [.05],
    "alpha": [50]
}
param_grid = generate_param_grid(param_dict)

print ("Num models to be tested: ", len(param_grid))

model_list=[]
for p in param_grid:
    als = ALS(userCol=COL_USER
          , itemCol=COL_ITEM
          , ratingCol=COL_RATING
          , coldStartStrategy="drop", nonnegative = True, implicitPrefs = True
          ,**p
          )
    model_list.append(als)
del p

# Cross Valid, evaluate also depands on K
cv_results = {
    'Model': [],

    'NDCG@K': [],
    'MAP@K': [],
    'Precision@K': [],
    'Recall@K': [],
    'cv_time': []
}

from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation, SparkDiversityEvaluation
from statistics import mean as stat_mean

def ranking_metrics_pyspark(test, predictions, k=TOP_K):
    rank_eval = SparkRankingEvaluation(
        test, predictions, k=k, relevancy_method="top_k", **COL_DICT
    )
    return {
        "NDCG@K": rank_eval.ndcg_at_k(),
        "MAP@K": rank_eval.map_at_k(),
        "Precision@K": rank_eval.precision_at_k(),
        "Recall@K": rank_eval.recall_at_k(),
    }

# Loops through all models and all folds
for count, model in enumerate(model_list, start=1):
    cv_results['Model'].append("ALS " + str(count))
    with Timer() as cv_time:
        cv_results_temp={
            'NDCG@K': [],
            'MAP@K': [],
            'Precision@K': [],
            'Recall@K': []
            }
        for ft_pair in foldlist:
            # Fits model to fold within training data
            train_model = model.fit(ft_pair[0])
            
            # Generates predictions using fitted_model on respective CV test data
            validation_predictions = train_model.transform(ft_pair[1]).drop(COL_RATING)
            
            # validation_performance = SparkRankingEvaluation(
            # ft_pair[1], 
            # validation_predictions,
            # col_user=COL_USER,
            # col_item=COL_ITEM,
            # col_rating=COL_RATING,
            # col_prediction=COL_PREDICTION,
            # k=TOP_K,
            # relevancy_method="top_k"
            # )
            # cv_results_temp['NDCG@K'].append(validation_performance.ndcg_at_k())
            # cv_results_temp['MAP@K'].append(validation_performance.map_at_k())
            # cv_results_temp['Precision@K'].append(validation_performance.precision_at_k())
            # cv_results_temp['Recall@K'].append(validation_performance.recall_at_k())
            
            dummy=ranking_metrics_pyspark(ft_pair[1],validation_predictions,k=TOP_K)

            cv_results_temp['NDCG@K'].append(dummy['NDCG@K'])
            cv_results_temp['MAP@K'].append(dummy['MAP@K'])
            cv_results_temp['Precision@K'].append(dummy['Precision@K'])
            cv_results_temp['Recall@K'].append(dummy['Recall@K'])
            
            del train_model, validation_predictions
            gc.collect()
            
    # cv_results['Top K'].append(validation_performance.k)
    cv_results['NDCG@K'].append(stat_mean(cv_results_temp['NDCG@K']))
    cv_results['MAP@K'].append(stat_mean(cv_results_temp['MAP@K']))
    cv_results['Precision@K'].append(stat_mean(cv_results_temp['Precision@K']))
    cv_results['Recall@K'].append(stat_mean(cv_results_temp['Recall@K']))
    cv_results['cv_time'].append(cv_time.interval)

del count, model

cv_results=pd.DataFrame.from_dict(cv_results) 

# Extract best combination of values from cross validation
best_model = model.bestModel

best_model = model_list[0]

for count, item in enumerate(model_list, start=1):
    
    print ("Model List: ", count)
    
    # Extract the Rank
    print ("Rank: ", item.getRank())
    
    # Extract the MaxIter value
    print ("MaxIter: ", item.getMaxIter())
    
    # Extract the RegParam value
    print ("RegParam: ", item.getRegParam())
    
    # Extract the Alpha value
    print ("Alpha: ", item.getAlpha())

results = {
    'train_time': [],
    'valid_time': [],
    'Model': [],
    # 'Top K': [],
    'NDCG@K': [],
    'MAP@K': [],
    'Precision@K': [],
    'Recall@K': []
}
    
# Loops through all models and all folds
for count, model in enumerate(model_list, start=1):
    with Timer() as train_time:
        # Fits model to all of training data and generates preds for test data
        train_model = model.fit(data_train_final)
    results['train_time'].append(train_time.interval)
    # train_time=train_time.interval
    # print("Took {} seconds for data_train_final.".format(train_time.interval))
    
    with Timer() as valid_time:
        validation_predictions = train_model.transform(data_valid_final).drop(COL_RATING)
        
        
        dummy=ranking_metrics_pyspark(data_valid_final, validation_predictions, k=TOP_K)
    results['Model'].append("ALS " + str(count))
    results['NDCG@K'].append(dummy['NDCG@K'])
    results['MAP@K'].append(dummy['MAP@K'])
    results['Precision@K'].append(dummy['Precision@K'])
    results['Recall@K'].append(dummy['Recall@K'])
    del dummy
    
    results['valid_time'].append(valid_time.interval)
    
    

results=pd.DataFrame.from_dict(results)      

# Remove seen items.
validation_predictions_exclude_train = validation_predictions.alias("pred").join(
    training.alias("train"),
    (validation_predictions[COL_USER] == training[COL_USER]) & (validation_predictions[COL_ITEM] == training[COL_ITEM]),
    how='outer'
)

validation_predictions_exclude_train = validation_predictions_exclude_train.filter(validation_predictions_exclude_train["train.Purchases"].isNull()) \
    .select('pred.' + COL_USER, 'pred.' + COL_ITEM, 'pred.' + "prediction")

from pyspark.sql.window import Window
import pyspark.sql.functions as F
window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())    
top_k_reco = validation_predictions_exclude_train.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")

a= top_k_reco.toPandas()

a2=a.describe()

a2=a.head(100)

def get_diversity_results(diversity_eval):
    metrics = {
        "catalog_coverage":diversity_eval.catalog_coverage(),
        "distributional_coverage":diversity_eval.distributional_coverage(), 
        "novelty": diversity_eval.novelty(), 
        "diversity": diversity_eval.diversity(), 
        "serendipity": diversity_eval.serendipity()
    }
    return metrics 

als_diversity_eval = SparkDiversityEvaluation(
    train_df = training, 
    reco_df = top_k_reco,
    col_user = COL_USER, 
    col_item = COL_ITEM
)

als_diversity_metrics = get_diversity_results(als_diversity_eval)
als_diversity_eval.catalog_coverage()
als_diversity_eval.distributional_coverage()
als_diversity_eval.novelty()
als_diversity_eval.diversity()
als_diversity_eval.serendipity()
        
a=validation_predictions.where("prediction>0").toPandas()
a_valid=validating.where("Purchases>0").toPandas()

check_cust_inner_1=a.copy()
check_cust_inner_2=a_valid.copy()
check_cust_inner_1=user_subset_recs.copy()
del a, a_valid
# Outer join check_cust_inner_1 and check_cust_inner_2
a=check_cust_inner_1.merge(check_cust_inner_2, on = ["CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID"], how = "outer", indicator = True)

a2=a.head(100)
a["_merge"].value_counts()
del a2
# Customer who bought many different products in descending order
a.loc[(a["Purchases"]>0)]\
    .groupby(["CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID"]).agg({"Purchases":"count"})\
    .groupby(["CUST_ID_Int"]).agg({"Purchases":"sum"})\
    .sort_values(["Purchases"], ascending = False)
    # [a["CUST_ID_Int"].isin(set(a[a["_merge"]=="right_only"]["CUST_ID_Int"]))]
    
    
a_dump=check_cust_inner_1.loc[(check_cust_inner_1["CUST_ID_Int"]==2424)]
a2=a.loc[(a["CUST_ID_Int"]==2424)]
a2=a.loc[(a["CUST_ID_Int"]==4499)]
a2=a.loc[(a["CUST_ID_Int"]==4781)]
# a2=a.loc[(a["CUST_ID_Int"]==2424) & (~a["prediction"].isnull())]
# Top n recommendations for all users
user_subset = validating.where("CUST_ID_Int=2424")
from pyspark.sql.functions import *
user_subset_recs = train_model.recommendForUserSubset(user_subset, 30)\
    .withColumn("itemIds_and_ratings", explode("recommendations"))\
    .select('CUST_ID_Int', col("itemIds_and_ratings.PROD_TITLE_PARENT_PROD_ID").alias("PROD_TITLE_PARENT_PROD_ID")\
    , col("itemIds_and_ratings.rating").alias("prediction")).toPandas()




als

model = als.fit(training)

predictions = model.transform(testing).cache()


# Define evaluator as RMSE 
evaluator = RankingEvaluator(k=TOP_K, metricName="ndcgAtK", labelCol=COL_RATING, predictionCol=COL_PREDICTION)


# Obtain and print RMSE
rmse = evaluator.evaluate(predictions)


# Cross Validation Step
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5, parallelism=5)         

model = cv.fit(training)













test_predictions = best_model.transform(testing)
test_performance = ROEM(test_predictions)

# Select some customers from training to check
check_cust=training.sort("Purchases", ascending=False).limit(500).select("CUST_ID_Int").distinct().cache()

# Select inner customers from test_predictions and check_cust to check
check_cust_inner = test_predictions.join(check_cust, "CUST_ID_Int", "inner").cache()

# Select info from training and test_predictions, pass to pandas to check
check_cust_inner_1 = training.join(check_cust_inner, "CUST_ID_Int", "leftsemi").toPandas()
check_cust_inner_2 = test_predictions.join(check_cust_inner, "CUST_ID_Int", "leftsemi").toPandas()

# Outer join check_cust_inner_1 and check_cust_inner_2
a=check_cust_inner_1.merge(check_cust_inner_2, on = ["CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID"], how = "outer", indicator = True)

# Customer who bought many different products in descending order
a.loc[(a["Purchases_x"]>0)]\
    .groupby(["CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID"]).agg({"Purchases_x":"count"})\
    .groupby(["CUST_ID_Int"]).agg({"Purchases_x":"sum"})\
    .sort_values(["Purchases_x"], ascending = False)\
    [a["CUST_ID_Int"].isin(set(a[a["_merge"]=="right_only"]["CUST_ID_Int"]))]
    
    

a2=a.loc[(a["CUST_ID_Int"]==3750) & (a["_merge"]=="right_only")]
a2["prediction_Percentile_Rank"]=a2["prediction"].rank(ascending = False, pct = True)
a2["item_rank"]=a2["Purchases_y"]*a2["prediction_Percentile_Rank"]


a_train_see2.loc[(a_train_see2["CUST_ID_Int"]==3750) & (a_train_see2["PROD_TITLE_PARENT_PROD_ID"]==1052813638)]
a_train_see2.info()

a2=a.loc[(a["CUST_ID_Int"]==11334) & (a["Purchases_x"]>0)]

a2=a.loc[(a["_merge"]=="right_only") & (a["prediction"]>0)].head(10)

a["_merge"].value_counts()

# Check training data' predictions
a_train_see=best_predictions.where("CUST_ID_Int = 3750").toPandas()
a_train_see2=a_train_see.copy()
a_train_see2["prediction_Percentile_Rank"]=a_train_see2["prediction"].rank(ascending = False, pct = True)
a_train_see2["item_rank"]=a_train_see2["Purchases"]*a_train_see2["prediction_Percentile_Rank"]
a_train_see2.loc[a_train_see2["item_rank"]>0].agg({"prediction_Percentile_Rank": ["min","max"]})["prediction_Percentile_Rank"].apply(lambda x:x*100).map('{:,.2f}%'.format)
# =============================================================================
# Percentage Range of how many purchased item successfully fall into the prediction_Percentile_Rank
# min    0.41%
# max    8.30%
# Name: prediction_Percentile_Rank, dtype: object
# e.g. One item first appears at 0.41% of all recommended items, one item appears lastly at 8.3% of all recommended items
# =============================================================================

a.columns

a.info()

test_predictions.createOrReplaceTempView("test_predictions_temp")

test_predictions_clean_recs = spark.sql("select CUST_ID_Int,\
                       itemIds_and_ratings.PROD_TITLE_PARENT_PROD_ID AS PROD_TITLE_PARENT_PROD_ID,\
                       itemIds_and_ratings.rating AS prediction\
                       from test_predictions_temp \
                       LATERAL VIEW explode(recommendations) exploded_table AS itemIds_and_ratings").cache()




from pyspark.ml.recommendation import ALSModel

# Top n recommendations for all users
user_subset = training.sort("Purchases", ascending=False).select("CUST_ID_Int").limit(30)
user_subset_recs = best_model.recommendForUserSubset(user_subset, 20)

# =============================================================================
# user_subset.select("CUST_ID_Int").distinct().join(user_subset_recs.select("CUST_ID_Int").distinct(), "CUST_ID_Int", "leftanti")
# 
# a=user_subset.select("CUST_ID_Int").distinct().toPandas()
# 
# a1=user_subset_recs.select("CUST_ID_Int").distinct().toPandas()
# 
# a2=a.merge(a1, how = "outer", on = "CUST_ID_Int", indicator = True)
# 
# df.loc[df['_merge']!='both']
# =============================================================================

# ALS_recommendations.count()
# training.select("CUST_ID_Int").distinct().count()

# SQL Method
user_subset_recs.createOrReplaceTempView("ALS_user_subset_recs_temp")

clean_recs = spark.sql("select CUST_ID_Int,\
                       itemIds_and_ratings.PROD_TITLE_PARENT_PROD_ID AS PROD_TITLE_PARENT_PROD_ID,\
                       itemIds_and_ratings.rating AS prediction\
                       from ALS_user_subset_recs_temp \
                       LATERAL VIEW explode(recommendations) exploded_table AS itemIds_and_ratings").cache()

# =============================================================================
# # Dataframe Method
# from pyspark.sql.functions import *
# clean_recs = user_subset_recs\
#     .withColumn("itemIds_and_ratings", explode("recommendations"))\
#     .select('CUST_ID_Int', col("itemIds_and_ratings.PROD_TITLE_PARENT_PROD_ID").alias("PROD_TITLE_PARENT_PROD_ID")\
#     , col("itemIds_and_ratings.rating").alias("prediction")).cache()
# =============================================================================

training.sort("Purchases", ascending=False).show(3)
+-------------------------+-----------+---------+
|PROD_TITLE_PARENT_PROD_ID|CUST_ID_Int|Purchases|
+-------------------------+-----------+---------+
|               1069686082|      12124|      537|
|               1090802790|      14688|      493|
|               1022978714|       8805|      330|
+-------------------------+-----------+---------+

from pyspark.sql.functions import *

# Add CUST_ID, Purchases column
clean_recs.join(cross_join, ["CUST_ID_Int", "PROD_TITLE_PARENT_PROD_ID"], "left").createOrReplaceTempView("cal_rank_ROEM")

# Cal ROEM
userCol="CUST_ID_Int"
cal_rank_ROEM=spark.sql("SELECT CUST_ID_Int, CUST_ID, PROD_TITLE_PARENT_PROD_ID, Purchases, prediction,\
            PERCENT_RANK() OVER (PARTITION BY " + userCol + " ORDER BY prediction DESC) AS PERCENT_RANK FROM cal_rank_ROEM")\
            .withColumn("np*rank", col("prediction")*col("PERCENT_RANK")).cache()

check_view=cal_rank_ROEM.where("CUST_ID_Int == 12124")

# Relevant Column for check_view
check_view_col=parqDF.select('PROD_TITLE_PARENT_PROD_ID', 'BRAND', 'PROD_CAT', 'PROD_TITLE_PARENT_TITLE').distinct().cache()

check_view=check_view.join(check_view_col, "PROD_TITLE_PARENT_PROD_ID", "left").toPandas()








# =============================================================================
# ROEM Metrics and Cross-Validation
# =============================================================================
best_validation_performance = 9999999999999
model_list = []

ranks = [10, 20]
maxIters = [10, 25]
regParams = [.05, .1]
alphas = [1, 30, 40, 50]

# Second Round
ranks = [10]
maxIters = [10]
regParams = [.05]
alphas = [50]

from pyspark.ml.recommendation import ALS
#Looping through each combindation of hyperparameters to ensure all combinations are tested.
for r in ranks:
  for mi in maxIters:
    for rp in regParams:
      for a in alphas:
        #Create ALS model
        model_list.append(ALS(userCol="CUST_ID_Int", itemCol="PROD_TITLE_PARENT_PROD_ID", ratingCol="Purchases", rank = r, maxIter = mi, regParam = rp, alpha = a,
                  coldStartStrategy="drop", nonnegative = True, implicitPrefs = True))
        

del r, mi, rp, a

# Print the model list, and the length of model_list
print (model_list, "Length of model_list: ", len(model_list))

# Validate
len(model_list) == (len(ranks)*len(maxIters)*len(regParams)*len(alphas))

# Empty list to fill with ROEMs from each model
ROEMS = []

#Expected percentile rank error metric function
def ROEM(predictions, userCol = "CUST_ID_Int", itemCol = "PROD_TITLE_PARENT_PROD_ID", ratingCol = "Purchases"):
    #Creates table that can be queried
    predictions.createOrReplaceTempView("predictions")
  
    #Sum of total number of plays of all songs
    denominator = predictions.groupBy().sum(ratingCol).collect()[0][0]
  
    #Calculating rankings of songs predictions by user
    spark.sql("SELECT " + userCol + " , " + ratingCol + " , PERCENT_RANK() OVER (PARTITION BY " + userCol + " ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")
  
    #Multiplies the rank of each song by the number of plays and adds the products together
    numerator = spark.sql('SELECT SUM(' + ratingCol + ' * rank) FROM rankings').collect()[0][0]
  
    performance = numerator/denominator
  
    return performance

# Loops through all models and all folds
for model in model_list:
    with Timer() as train_time:
        # Fits model to all of training data and generates preds for test data
        validation_model = model.fit(training)
    print("Took {} seconds for training.".format(train_time.interval))
    validation_predictions = validation_model.transform(validating)
    validation_performance = ROEM(validation_predictions)
    print ("Validation ROEM: ", validation_performance)

    # Adds validation ROEM to ROEM list
    ROEMS_v=[]
    ROEMS_v.append(validation_performance)
    
    ROEMS_cf=[]
    #Filling in final hyperparameters with those of the best-performing model
    if validation_performance < best_validation_performance:
        best_model = validation_model
        best_predictions = validation_predictions
        best_validation_performance = validation_performance
        
        for ft_pair in foldlist:
        # Fits model to fold within training data
          fitted_model = model.fit(ft_pair[0])
      
          # Generates predictions using fitted_model on respective CV test data
          predictions = fitted_model.transform(ft_pair[1])
      
          # Generates and prints a ROEM metric CV test data
          ROEM_cf = ROEM(predictions)
          
          ROEMS_cf.append(ROEM_cf)
    else:
        pass
    ROEMS.append([ROEMS_cf, ROEMS_v])
    
del validation_model, validation_predictions, validation_performance
del fitted_model, predictions, ROEM_cf

for count, item in enumerate(model_list, start=1):
    if item.uid == best_model.uid:
        print ("Model List: ", count)
        
        print ("ROEM validation: ", ROEMS[count-1][1])
        print ("ROEM cv: ", ROEMS[count-1][0])
        
        # Extract the Rank
        print ("Rank: ", item.getRank())
        
        # Extract the MaxIter value
        print ("MaxIter: ", item.getMaxIter())
        
        # Extract the RegParam value
        print ("RegParam: ", item.getRegParam())
        
        # Extract the Alpha value
        print ("Alpha: ", item.getAlpha())
    else:
        pass
        
del count, item






# =============================================================================
# Pandas, Arrow
# =============================================================================

os.chdir("//hk-nas05/Business_Planning/01_Analysis/A0025_KR-Beauty Brand Gross Margin/Output/")
raw_0583_h1=pd.read_parquet("A0025_result_raw_(2021H1)_2022-02-21.parquet")
raw_0583_h2=pd.read_parquet("A0025_result_raw_(2021H2)_2022-02-21.parquet")
raw_0583=pd.concat([raw_0583_h1, raw_0583_h2], ignore_index=True)
del raw_0583_h1, raw_0583_h2
# a=pd.to_parquet("abc.parquet")

raw_0583.head()
raw_0583.info()

reduced_df = reduce_memory_usage(raw_0583, verbose = True)
del raw_0583
# Convert to datetime
reduced_df[reduced_df.filter(regex="_DT$").columns]=reduced_df.filter(regex="_DT$").apply(lambda x: pd.to_datetime(x, format= '%Y-%m-%d %H:%M:%S') if x.name in ["INVOICE_DT", "ORDER_DT"] else 
                                      pd.to_datetime(x, format= '%d/%m/%Y')
                                      , axis = 0)
reduced_df.filter(regex="_DT$").info()
reduced_df.filter(regex="_DT$").head()
reduced_df[reduced_df["INVOICE_DT"].isna()]

# Convert to datetime
reduced_df.loc[:, reduced_df.columns.str.contains("date", case=False, regex=True)] = reduced_df.loc[:, reduced_df.columns.str.contains("date", case=False, regex=True)].apply(lambda x: pd.to_datetime(x, format= '%Y-%m-%d %H:%M:%S')
                                      , axis = 0)

# Convert to string
reduced_df.loc[:, [col for col in reduced_df.columns if reduced_df[col].dtype == "object"]] = reduced_df.loc[:, [col for col in reduced_df.columns if reduced_df[col].dtype == "object"]].apply(lambda x: x.astype('string'), axis=0)

os.chdir("//hk-nas05/Business_Planning/03_Regular_Report/R0001 - YS Shipping Rate/Input/")
cust_mas=dt.fread("dm_t_order_01-12.csv").to_pandas()
cust_mas=cust_mas.astype("string")
cust_mas.info()

gc.collect()

reduced_df=reduced_df.merge(cust_mas, how='left', on=['BE_ORDER_ID'], indicator=True, copy=False)

reduced_df=reduced_df[reduced_df["_merge"]=="both"]


del cust_mas
gc.collect()

reduced_df.info()

reduced_df.value_counts(["BE_ORDER_ID", "SKU"])

####################################################################################
from sklearn.neighbors import NearestNeighbors

us_order=reduced_df[reduced_df['SHIP_TO_COUNTRY']=="United States"].copy()
del reduced_df
basket_us = (us_order
          .groupby(['BE_ORDER_ID', 'SKU'])['PROD_QTY']
          .sum().unstack().reset_index().fillna(0)
          .set_index('BE_ORDER_ID'))

gc.collect()

a=us_order.groupby(['BE_ORDER_ID', 'SKU'])['PROD_QTY'].sum().unstack().reset_index().fillna(0).set_index('BE_ORDER_ID')
a.describe()

us_order.info()

us_order.value_counts(["SKU"]).describe()






