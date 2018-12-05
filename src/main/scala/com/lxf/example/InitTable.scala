package com.lxf.example

import com.lxf.example.InitTable.DouPan
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object InitTable {
  case class DouPan(userid:Int,username:String,movieid:Int,rating:Double)

  def main(args: Array[String]): Unit = {
    val textFile: String = checkParms(args)
    SetLogger
    val (spark, sparkContext) = SetSpark
    val douPanDf = saveTable2Hive(spark, textFile, sparkContext)

    import spark.sql
    //统计评分次数最多的电影
    sql("select movieid,count(*) from doupan2 group by movieid limit 10")
  }




  private def checkParms(args: Array[String]) = {
    if (args.length != 1) {
      System.err.println("Request Parameters :<textFile> <fileSavePath>")
      System.exit(1)
    }
    val textFile = args(0)
    textFile
  }

  private def saveTable2Hive(spark: SparkSession, textFile: String, sparkContext: SparkContext) = {
    import spark.implicits._
    import spark.sql
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


    //将处理后的数据存入hive中，方便读取
    val douPanDf=name2id.join(preRdd).map{
      //(uname,uid,gid,rate)
      case (k,(v1,v2))=>DouPan(v1.toInt,k,v2._1.toInt,v2._2.toDouble)
    }.toDF().createTempView("doupandf")
    sql("insert into doupan2 select * from doupandf")
    douPanDf
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
