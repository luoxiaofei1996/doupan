package com.lxf.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.{DateTime, Duration}

object AlsEvaluation {

  def main(args: Array[String]) {
    if (args.length != 1) {
      System.err.print("Request Parameter : <textFile>")
      System.exit(1)
    }
    val textFile=args(0)
    //SetLogger
    println("==========数据准备阶段===============")
    val (trainData, validationData, testData) = PrepareData(textFile)
    trainData.persist(); validationData.persist(); testData.persist()
    println("==========训练验证阶段===============")
    val bestModel = trainValidation(trainData, validationData)
    println("==========测试阶段===============")
    val testRmse = computeRMSE(bestModel, testData)
    println("使用testData测试bestModel," + "结果rmse = " + testRmse)
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def trainValidation(trainData: RDD[Rating], validationData: RDD[Rating]): MatrixFactorizationModel = {
    println("-----评估 rank参数使用 ---------")
    evaluateParameter(trainData, validationData, "rank", Array(3,4,5, 6,7), Array(25), Array(0.1))
    println("-----评估 numIterations ---------")
    evaluateParameter(trainData, validationData, "numIterations", Array(5), Array(21,23,25,27,29), Array(0.1))
    println("-----评估 lambda ---------")
    evaluateParameter(trainData, validationData, "lambda", Array(5), Array(25), Array(0.08,0.09,0.1,0.11,0.12))
    println("-----所有参数交叉评估找出最好的参数组合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array(3,4,5, 6,7), Array(21,23,25,27,29), Array(0.08,0.09,0.1,0.11,0.12))
    return (bestModel)
  }
  def evaluateParameter(trainData: RDD[Rating], validationData: RDD[Rating],
                        evaluateParameter: String, rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double]) =
    {
      for (rank <- rankArray; numIterations <- numIterationsArray; lambda <- lambdaArray) {
        val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)
      }
    }

  def evaluateAllParameter(trainData: RDD[Rating], validationData: RDD[Rating],
                           rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double]): MatrixFactorizationModel =
    {
      val evaluations =
        for (rank <- rankArray; numIterations <- numIterationsArray; lambda <- lambdaArray) yield {
          val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)

          (rank, numIterations, lambda, rmse)
        }
      val Eval = (evaluations.sortBy(_._4))
      val BestEval = Eval(0)
      println("最佳model参数：rank:" + BestEval._1 + ",iterations:" + BestEval._2 + "lambda" + BestEval._3 + ",结果rmse = " + BestEval._4)
      val bestModel = ALS.train(trainData, BestEval._1, BestEval._2, BestEval._3)
      (bestModel)
    }
  def PrepareData(textFile:String): (RDD[Rating], RDD[Rating], RDD[Rating]) = {

    val sc = new SparkContext(new SparkConf()
//      .setAppName("RDF")
//      .setMaster("local[*]")
//      .set("spark.executor.memory", "6g")
//      .set("spark.driver.memory","6g")
    )
    //----------------------1.创建用户评分数据-------------

    print("开始读取用户评分数据...")

    val preRdd = sc.textFile(textFile)
      .map(_.split("::"))
      .map(u => (u(0), (u(1), u(2))))
    val name2id = preRdd
      .map(_._1)
      .distinct()
      .zipWithUniqueId()
      .map(u => (u._1.toString, u._2.toString))

    val rawRatings = preRdd.join(name2id).map{
      case (key,((movie,rating),user)) =>Array(user,movie,rating)
    }

    val ratingsRDD = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    println("共计：" + ratingsRDD.count.toString() + "条ratings")


    //----------------------3.显示数据记录数-------------
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("共计：ratings: " + numRatings + " User " + numUsers + " Movie " + numMovies)
    //----------------------4.以随机方式将数据分为3个部分并且返回-------------
    println("将数据分为")
    val Array(trainData, validationData, testData) = ratingsRDD.randomSplit(Array(0.8, 0.1, 0.1))

    println("  trainData:" + trainData.count() + "  validationData:" + validationData.count() + "  testData:" + testData.count())
    return (trainData, validationData, testData)
  }

  def trainModel(trainData: RDD[Rating], validationData: RDD[Rating], rank: Int, iterations: Int, lambda: Double): (Double, Double) = {
    val startTime = new DateTime()
    val model = ALS.train(trainData, rank, iterations, lambda)
    val endTime = new DateTime()
    val Rmse = computeRMSE(model, validationData)
    val duration = new Duration(startTime, endTime)
    println(f"训练参数：rank:$rank%3d,iterations:$iterations%.2f ,lambda = $lambda%.4f 结果 Rmse=$Rmse%.4f" + "训练花费时间" + duration.getMillis + "毫秒")
    (Rmse, duration.getStandardSeconds)
  }

  def computeRMSE(model: MatrixFactorizationModel, RatingRDD: RDD[Rating]): Double = {

    val num = RatingRDD.count()
    val predictedRDD = model.predict(RatingRDD.map(r => (r.user, r.product)))
    val predictedAndRatings =
      predictedRDD.map(p => ((p.user, p.product), p.rating))
        .join(RatingRDD.map(r => ((r.user, r.product), r.rating)))
        .values
    math.sqrt(predictedAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / num)
  }

  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

}
