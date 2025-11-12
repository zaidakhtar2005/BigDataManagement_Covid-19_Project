#!/usr/bin/env python3
# Big Data Management Project — COVID-19 (Kaggle)
# - Kaggle CSV dataset (covid_19_data.csv)
# - 3 batch queries in Apache Spark
# - Experimental evaluation (execution times)
# - Export outputs to a NoSQL store (MongoDB)

import time
from statistics import mean

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, to_date, coalesce, lit, lag,
    when, max as _max, sum as _sum, row_number, desc, date_format
)

# For Mongo export sanitation
import pandas as pd

# =========================
# CONFIG — EDIT AS NEEDED
# =========================
CSV_PATH = "data/covid_19_data.csv"

# Clamp negative daily deltas to 0? (helps when source revises cumulative counts downward)
CLAMP_NEGATIVE_DELTAS = True

# MongoDB export (NoSQL requirement)
EXPORT_TO_MONGO = True
MONGO_URI = "mongodb+srv://zaidakhtar191123_db_user:bKoVPpCsqJUbrYkU@cluster0.crkaxum.mongodb.net/"
MONGO_DB = "covid_db"
MONGO_LIMIT_DOCS = 20000  # avoid huge uploads; raise if you want
MONGO_CHUNK_SIZE = 5000   # insert in chunks to avoid big payloads

# Spark resources (you can tweak if needed)
SPARK_DRIVER_MEM = "4g"
SPARK_MASTER = "local[*]"

# =========================
# 0) START SPARK
# =========================
spark = SparkSession.builder \
    .master(SPARK_MASTER) \
    .appName("Covid19SparkProject") \
    .config("spark.driver.memory", SPARK_DRIVER_MEM) \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print(">> Spark version:", spark.version)

# =========================
# 1) LOAD + CLEAN
# =========================
# Expected columns in Kaggle file:
# SNo, ObservationDate, Province/State, Country/Region, Last Update, Confirmed, Deaths, Recovered
df_raw = spark.read.option("header", True).option("inferSchema", True).csv(CSV_PATH)

df = (
    df_raw
    # Parse mm/dd/yyyy to date
    .withColumn("date", to_date(col("ObservationDate"), "MM/dd/yyyy"))
    # Normalize country name (optional; keeps "Mainland China" as "China")
    .withColumn("country",
                when(col("Country/Region") == "Mainland China", "China")
                .otherwise(col("Country/Region")))
    # Province can be null; keep as-is (window partitions will handle nulls fine)
    .withColumn("province", col("Province/State"))
    # Ensure numeric types & fill nulls with 0
    .withColumn("confirmed", coalesce(col("Confirmed").cast("long"), lit(0)))
    .withColumn("deaths",    coalesce(col("Deaths").cast("long"),    lit(0)))
    .withColumn("recovered", coalesce(col("Recovered").cast("long"), lit(0)))
    .select("date", "country", "province", "confirmed", "deaths", "recovered")
    .dropna(subset=["date", "country"])
)

print("\n✅ Sample after cleaning:")
df.show(5, truncate=False)

# =========================
# 2) DAILY SNAPSHOT (IMPORTANT)
#    Collapse intra-day duplicates: one row per (date, country, province)
#    Take MAX of cumulative counters for that day (last update of the day).
# =========================
daily = (
    df.groupBy("date", "country", "province")
      .agg(
          _max("confirmed").alias("confirmed"),
          _max("deaths").alias("deaths"),
          _max("recovered").alias("recovered"),
      )
)

# =========================
# 3) QUERY 1 — Daily increment per province/state
# =========================
w = Window.partitionBy("country", "province").orderBy("date")
df_q1 = (
    daily
    .withColumn("prev_confirmed", coalesce(lag("confirmed").over(w), lit(0)))
    .withColumn("daily_new_confirmed", col("confirmed") - col("prev_confirmed"))
)

if CLAMP_NEGATIVE_DELTAS:
    df_q1 = df_q1.withColumn(
        "daily_new_confirmed",
        when(col("daily_new_confirmed") < 0, 0).otherwise(col("daily_new_confirmed"))
    )

df_q1 = df_q1.select(
    "date", "country", "province", "confirmed", "prev_confirmed", "daily_new_confirmed"
)

print("\n=== Query 1 — preview ===")
df_q1.orderBy("date", "country", "province").show(10, truncate=False)

# =========================
# 4) QUERY 2 — Top 5 countries per day by new infections
#     Sum province-level deltas to country-level, rank per day.
# =========================
country_daily = (
    df_q1.groupBy("date", "country")
         .agg(_sum("daily_new_confirmed").alias("new_cases_country"))
)
w2 = Window.partitionBy("date").orderBy(desc("new_cases_country"))
df_q2 = (
    country_daily
    .withColumn("rank", row_number().over(w2))
    .where(col("rank") <= 5)
    .select("date", "country", "new_cases_country", "rank")
)

print("\n=== Query 2 — preview ===")
df_q2.orderBy("date", "rank").show(10, truncate=False)

# =========================
# 5) QUERY 3 — Recovery-to-death ratio per country per day
#     Use daily snapshot, aggregate to country per day, safe divide.
# =========================
df_q3 = (
    daily.groupBy("date", "country")
         .agg(
             _sum("recovered").alias("recovered"),
             _sum("deaths").alias("deaths")
         )
         .withColumn(
             "recov_death_ratio",
             when(col("deaths") == 0, None).otherwise(col("recovered") / col("deaths"))
         )
         .select("date", "country", "recovered", "deaths", "recov_death_ratio")
)

print("\n=== Query 3 — preview ===")
df_q3.orderBy("date", "country").show(10, truncate=False)

# =========================
# 6) PERFORMANCE EVALUATION (Batch)
#     Measure each query 3 times (count triggers compute) and print averages.
# =========================
def time_df(label, df_):
    runs = []
    for _ in range(3):
        t0 = time.time()
        _ = df_.count()
        runs.append(time.time() - t0)
    avg = mean(runs)
    print(f"{label}: runs={runs} | avg={avg:.3f}s")
    return avg, runs

print("\n=== Measuring batch execution times ===")
t1_avg, t1_runs = time_df("Query 1", df_q1)
t2_avg, t2_runs = time_df("Query 2", df_q2)
t3_avg, t3_runs = time_df("Query 3", df_q3)

print("\nAverage times (s): Q1={:.3f}  Q2={:.3f}  Q3={:.3f}".format(t1_avg, t2_avg, t3_avg))

# =========================
# 7) SAVE OUTPUTS (CSV)
# =========================
df_q1.write.mode("overwrite").csv("output/query1", header=True)
df_q2.write.mode("overwrite").csv("output/query2", header=True)
df_q3.write.mode("overwrite").csv("output/query3", header=True)
print("\n✅ Saved outputs to output/query1, output/query2, output/query3")

# =========================
# 8) OPTIONAL: EXPORT TO MONGODB (NoSQL) — with sanitation
#    - date -> string (yyyy-MM-dd)
#    - no NaN/NaT (replace with None)
#    - chunked inserts
# =========================
if EXPORT_TO_MONGO:
    import pymongo
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    # Make Mongo-safe DataFrames: stringify date, cast numerics
    df_q1_m = df_q1.select(
        date_format(col("date"), "yyyy-MM-dd").alias("date"),
        col("country"),
        col("province"),
        col("confirmed").cast("long").alias("confirmed"),
        col("prev_confirmed").cast("long").alias("prev_confirmed"),
        col("daily_new_confirmed").cast("long").alias("daily_new_confirmed"),
    )

    df_q2_m = df_q2.select(
        date_format(col("date"), "yyyy-MM-dd").alias("date"),
        col("country"),
        col("new_cases_country").cast("long").alias("new_cases_country"),
        col("rank").cast("int").alias("rank"),
    )

    df_q3_m = df_q3.select(
        date_format(col("date"), "yyyy-MM-dd").alias("date"),
        col("country"),
        col("recovered").cast("long").alias("recovered"),
        col("deaths").cast("long").alias("deaths"),
        col("recov_death_ratio").cast("double").alias("recov_death_ratio"),
    )

    def to_mongo(df_spark, collection_name, limit_docs=None, chunk_size=MONGO_CHUNK_SIZE):
        coln = db[collection_name]
        if limit_docs is not None:
            df_spark = df_spark.limit(limit_docs)

        # Convert to pandas once; replace NaN/NaT with None
        pdf = df_spark.toPandas()
        pdf = pdf.where(pd.notnull(pdf), None)

        total = 0
        for start in range(0, len(pdf), chunk_size):
            batch = pdf.iloc[start:start+chunk_size].to_dict("records")
            if batch:
                coln.insert_many(batch)
                total += len(batch)
        return total

    print("\n=== Exporting to MongoDB (Atlas) ===")
    n1 = to_mongo(df_q1_m, "query1", limit_docs=MONGO_LIMIT_DOCS)
    n2 = to_mongo(df_q2_m, "query2", limit_docs=MONGO_LIMIT_DOCS)
    n3 = to_mongo(df_q3_m, "query3", limit_docs=MONGO_LIMIT_DOCS)
    print(f"✅ MongoDB Atlas export done — query1={n1}, query2={n2}, query3={n3} docs inserted.")

print("\n🎉 All tasks completed successfully.")
spark.stop()
