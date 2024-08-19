
import random
import datetime
import uuid
from cassandra.cluster import Cluster
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from operator import add
from pyspark.sql import SQLContext

# Initialize Spark Context and Streaming Context
sc = SparkContext(appName="SmartParking")
ssc = StreamingContext(sc, 0.250)
sqlContext = SQLContext(sc)

# Initialize Cassandra Cluster and Session
cluster = Cluster()
session = cluster.connect('test')

totalSlots = 500.0
timestamp = datetime.datetime.utcnow()

def process_rdd(rdd):
    global timestamp
    l = rdd.collect()
    if len(l) != 3:
        return
    for lot, cars in l:
        lotid = lot.strip("'")
        occrate = (cars / totalSlots) * 100
        if occrate > 100:
            occrate = 100
        price = 2 + (occrate / 100) * 20 if occrate <= 100 else -1
        
        session.execute(
            "INSERT INTO smartpark (key, lotid, occrate, time, price) VALUES (%s, %s, %s, %s, %s)",
            [uuid.uuid4(), int(lotid), occrate, str(timestamp)[:-7], price]
        )
        
        seconds = random.randint(1800, 2400)
        timestamp += datetime.timedelta(seconds=seconds)

kvs = KafkaUtils.createStream(ssc, "ip-172-31-39-49.ec2.internal:2181", "spark-streaming-consumer", {'smartparking': 1})
lines = kvs.map(lambda x: x[1])
lines = lines.map(lambda line: line.encode('ascii', 'ignore'))
lines = lines.map(lambda line: line.split(","))
lines = lines.map(lambda line: (line[0].strip("'"), 1 if line[3].strip() == "True" else 0))
lines = lines.reduceByKey(add)
lines.foreachRDD(lambda rdd: process_rdd(rdd))

ssc.start()
ssc.awaitTermination()
