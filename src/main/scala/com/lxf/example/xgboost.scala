package com.lxf.example

import com.lxf.example.MovieCluster.SetLogger
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostModel}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer

object xgboost {
  def main(args: Array[String]): Unit = {

    /**
      * train XGBoost model with the DataFrame-represented data
      *  trainingData the trainingset represented as DataFrame
      *  params Map containing the parameters to configure XGBoost
      *  round the number of iterations
      *  nWorkers the number of xgboost workers, 0 by default which means that the number of
      *                 workers equals to the partition number of trainingData RDD
      *  obj the user-defined objective function, null by default
      *  eval the user-defined evaluation function, null by default
      *  useExternalMemory indicate whether to use external memory cache, by setting this flag as
      *                           true, the user may save the RAM cost for running XGBoost within Spark
      * missing the value represented the missing value in the dataset
      * featureCol the name of input column, "features" as default value
      *  labelCol the name of output column, "label" as default value
      */

    SetLogger
    val conf=new SparkConf().setMaster("local")
    val spark=SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.read.format("parquet").load("/Users/david/IdeaProjects/doupan/moviefeatures")
      .createTempView("moviefeatures")

    var array=ArrayBuffer[String]()
    for (i <- 0 to 39){
      array+="feature["+i+"]"
    }
    //使用sparksql取出movie的所有特征
    var sql= "select movie,"+array.mkString(",")+" from moviefeatures"
    val movieFeatures=spark.sql(sql)
    movieFeatures.show()

    val vecDF=new VectorAssembler().setInputCols(array.toArray).setOutputCol("feature").transform(movieFeatures)


    val maxDepth = 10
    val numRound = 10
    val nworker = 10
    val paramMap = List(
      "eta" -> 0.01, //学习率
      "gamma" -> 0.1, //用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
      "lambda" -> 2, //控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
      "subsample" -> 0.8, //随机采样训练样本
      "colsample_bytree" -> 0.8, //生成树时进行的列采样
      "max_depth" -> maxDepth, //构建树的深度，越大越容易过拟合
      "min_child_weight" -> 5,
      "objective" -> "multi:softprob",  //定义学习任务及相应的学习目标
      "eval_metric" -> "merror",
      "num_class" -> 21
    ).toMap

    val model:XGBoostModel = XGBoost.trainWithDataFrame(vecDF, paramMap, numRound, nworker,
      useExternalMemory = true,
      featureCol = "features",
      labelCol = "label",
      missing = 0.0f)

    //predict the test set
    val predict:DataFrame = model.transform(vecDF)

  }
}
