package com.lxf.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object CreateModel {

  //推荐
  case class Recommendation(id: Int, r: Double)

  // 用户的推荐
  case class UserRecs(uid: Int, recs: Seq[Recommendation])

  //电影的相似度
  case class MovieRecs(mid: Int, recs: Seq[Recommendation])

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      System.err.println("Request Parameters :<textFile> <modelSavePath>")
      System.exit(1)
    }
    SetLogger
    println("==========数据准备阶段===============")
    val (rank, iterations, lambda) = ((50, 5, 0.01))
    val (textFile, modelSavePath) = (args(0), args(1))
    //矩阵拆分时特征值的个数，训练轮次，
    val (spark: SparkSession, sparkContext: SparkContext) = setSpark
    val trainRdd: RDD[Rating] = prepareData(spark, textFile, sparkContext)
    println("==========模型训练阶段===============")
    /**
      * 训练ALS模型
      * ALS算法是协同过滤算法中的优化算法
      * 思想是：将用户/商品矩阵拆分为两个矩阵，矩阵1为用户/特征矩阵 ，矩阵2为电影/特征矩阵，其中的特征是抽象的概念
      * 电影id对应的特征值集合组成该电影的特征向量，用户id对应的特征值集合组成该用户的特征向量
      * 用户/电影之间的相似度可以由计算他们特征向量的欧式距离或余弦相似度得出
      */
    val model = ALS.train(trainRdd, rank, iterations, lambda)
    //将model保存到本地
    model.save(sparkContext, modelSavePath)
    println("==========计算用户推荐矩阵===============")
    computeUserRecMovies(spark,trainRdd, model)

    spark.stop()
  }


  private def computeUserRecMovies(spark:SparkSession,trainRdd: RDD[Rating], model: MatrixFactorizationModel) = {
    /**
      * 计算用户推荐矩阵
      * 需传入参数：RDD[(Int, Int)]
      */
      import spark.implicits._
    val usersProducts = trainRdd.map(u => (u.user, u.product))
    /**
      * 得到用户商品矩阵的全部信息
      * 转化为(user,(product,rate))的形式
      * 对用户进行group
      * 对每个用户所匹配的商品进行排序，取前10个
      */
    val userRecs = model.predict(usersProducts)
      .map(rating => (rating.user, (rating.product, rating.rating)))
      .groupByKey()
      .map {
        case (u, recs) => UserRecs(u, recs.toList.sortWith(_._2 > _._2).take(10).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()
    userRecs.show(false)
  }

  private def prepareData(spark: SparkSession, textFile: String, sparkContext: SparkContext) = {
    import spark.implicits._
    /**
      * 创建训练集
      * 格式；(user_name,movie_id,rating)
      */
    val preRdd = sparkContext.textFile(textFile)
      .map(_.split("::"))
      .map(u => (u(0), (u(1), u(2))))
    /**
      * name中有非数字字符，将其映射为数字
      * 并保存
      */
    val name2id = preRdd
      .map(_._1)
      .distinct()
      .zipWithUniqueId()
      .map(u => (u._1.toString, u._2.toString))

    name2id.toDF().write.format("parquet").mode("overwrite").save("file:////Users/david/IdeaProjects/doupan/name2id")


    val trainRdd = preRdd.join(name2id).map {
      case (key, ((movie, rating), user)) => Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    trainRdd.toDF().show()
    trainRdd
  }

  private def setSpark = {
    val conf = new SparkConf()
      .setMaster("local[4]")
      .set("spark.executor.memory", "10g")
      .set("spark.driver.memory", "6g")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("WARN")
    (spark, sparkContext)
  }

  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}
