package com.lxf.example

import com.lxf.example.CreateModel.{MovieRecs, Recommendation}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

object MoviesSimiliarity {

  case class MovieFeatures(movie:Int,feature: Array[Double])
  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      System.err.print("Request Parameter : <modelPath>")
      System.exit(1)
    }
    val modelPath = args(0)
    val conf = new SparkConf()
      .setMaster("local[7]")
      .set("spark.executor.memory", "10g")
      .set("spark.driver.memory", "6g")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("WARN")

    import spark.implicits._
    /**
      * 计算用户推荐矩阵
      * 形式如下：
      * i1  i2  i3
      * u1  2   2   2
      * u2
      * u3
      * 为了方便表示，记录方式变为 （u1,(i1,2)）,即为笛卡尔乘积
      */


    val model = MatrixFactorizationModel.load(sparkContext, modelPath)

    val moviefeatures = model.productFeatures.map {
      case (movie, features) => (movie, new DoubleMatrix(features))
    }.repartition(10)
      .cache()



    val array = sparkContext.broadcast(moviefeatures.collect().toList)

    moviefeatures.map{
      case (a,b)=>MovieFeatures(a,b.toArray)
    }.toDF().write.format("parquet").mode("overwrite").save("/Users/david/IdeaProjects/doupan/moviefeatures")

    //join操作转化为map操作，提高效率，节约内存（只有在大表join小表的时候有效）
    moviefeatures.map {
      case (movie, feature) => {
        val recs = array.value.filter { case (a, b) => a != movie }
          .map {
            case (a, b) => (a, consinSim(feature, b))
          }.filter(_._2 > 0.6)
          .take(10)
          .map(x => Recommendation(x._1, x._2))
        (movie, recs.toList)
      }
    }
      .map {
        case (a, b) => MovieRecs(a, b)
      }.toDF().show()



  }

  //计算两个电影之间的余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }

}
