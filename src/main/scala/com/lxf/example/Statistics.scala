package com.lxf.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Statistics {
  case class DouPan(userid:Int,username:String,movieid:Int,rating:Double)

  def main(args: Array[String]): Unit = {
    //SetLogger
    val (spark, sparkContext) = SetSpark

    import spark.sql
    //统计评分次数最多的电影
    println("----统计评分次数最多的电影----")
    sql("select movieid,count(*) count from doupan2 group by movieid  order by count desc limit 10").show()

    //统计平均分最高的电影
    println("----统计平均分最高的电影----")
    sql("select movieid ,avg(rating) rating from doupan2 group by movieid order by rating desc limit 10").show()

    println("----统计最活跃的用户（评论次数最多）----")
    sql("select username ,count(*) count from doupan2 group by username order by count desc limit 10").show()

    println("----统计潜在的恶意评分用户----")
    sql("select username ,avg(rating) rating,count(*) count" +
      " from doupan2 " +
      " group by username order by rating  limit 10").show()

  }

  private def SetSpark = {
    val conf = new SparkConf()
      .setMaster("local[4]")
      .set("spark.executor.memory", "10g")
      .set("spark.driver.memory", "6g")

    val spark = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()
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
