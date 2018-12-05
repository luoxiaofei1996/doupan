package com.lxf.example

import breeze.numerics.sqrt
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ALSTrainer {
  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      System.err.println("Request Parameter : <textFile>")
      System.exit(1)
    }
    SetLogger

    val textFile = args(0)
    //创建SparkConf
    val sparkConf = new SparkConf().setAppName("ALSTrainer").setMaster("local[6]")

    //创建SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("WARN")
    //加载数据
    /**
      * 创建训练集
      * 格式；(user_name,movie_id,rating)
      */
    val preRdd = sparkContext.textFile(textFile)
      .map(_.split("::"))
      .map(u => (u(0), (u(1), u(2))))
    /**
      * 为用户分配id
      * 格式：(user_id ,user_name)
      */
    val name2id = preRdd
      .map(_._1)
      .distinct()
      .zipWithUniqueId()
      .map(u => (u._1.toString, u._2.toString))

    val trainRdd = preRdd.join(name2id).map {
      case (key, ((movie, rating), user)) => Rating(user.toInt, movie.toInt, rating.toDouble)
    }.cache()

    //输出最优参数
    adjustALSParams(trainRdd)

    //关闭Spark
    spark.close()

  }

  // 输出最终的最优参数
  def adjustALSParams(trainData: RDD[Rating]): Unit = {
    val result = for (rank <- Array(30, 40, 50, 60, 70); iteration<-Array(5,10,15,20,25,30);lambda <- Array(1, 0.1, 0.001))
      yield {
        val model = ALS.train(trainData, rank, iteration, lambda)
        val rmse = getRmse(model, trainData)
        println(s"""目前测试参数：(rank:$rank, iteration:$iteration,lambda:$lambda)=>rmse:$rmse""")
        (rank, iteration,lambda, rmse)
      }
    println("最优参数(rank, iteration,lambda, rmse):"+result.sortBy(_._4).head)
  }

  def getRmse(model: MatrixFactorizationModel, trainData: RDD[Rating]): Double = {
    //需要构造一个usersProducts  RDD[(Int,Int)]
    val userMovies = trainData.map(item => (item.user, item.product))
    val predictRating = model.predict(userMovies)

    val real = trainData.map(item => ((item.user, item.product), item.rating))
    val predict = predictRating.map(item => ((item.user, item.product), item.rating))

    sqrt(
      real.join(predict).map { case ((uid, mid), (real, pre)) =>
        // 真实值和预测值之间的两个差值
        val err = real - pre
        err * err
      }.mean()
    )
  }

  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

}
