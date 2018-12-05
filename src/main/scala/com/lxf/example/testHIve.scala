package com.lxf.example

import org.apache.spark.sql.SparkSession

object testHIve {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .enableHiveSupport()
      .appName("Spark Hive")
      .master("local[2]")
      .getOrCreate()
    import spark.sql
    sql("use tpch")
    sql("show tables").show()
    sql("select count(*) from lineitem").show()
  }

}
