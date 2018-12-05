package com.lxf.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

object MovieCluster {
  case class MovieFeatures(movie:Int,feature: Array[Double])
  def main(args: Array[String]): Unit = {
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

    val features=new VectorAssembler().setInputCols(array.toArray).setOutputCol("feature").transform(movieFeatures)
    val Array(train , test)=features.randomSplit(Array(0.8,0.2))
    val model=new KMeans().setMaxIter(5).setFeaturesCol("feature").setPredictionCol("prediction").setK(5).fit(train)
    val result = model.transform(test)
    result.select("movie","feature","prediction").show()



  }

  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}
