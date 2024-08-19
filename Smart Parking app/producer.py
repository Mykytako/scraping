
import random
import datetime
from kafka import KafkaProducer

# Initialize Kafka Producer
producer = KafkaProducer(bootstrap_servers='ip-172-31-39-49.ec2.internal:6667')

class SensorMessage:
    def __init__(self, lotId, spotId, timeStamp, occupied):
        self.lotId = lotId
        self.spotId = spotId
        self.timeStamp = timeStamp
        self.occupied = occupied

def startParkingSession(lotId, spotId, timestamp):
    return SensorMessage(lotId, spotId, timestamp, True)

def endParkingSession(lotId, spotId, timestamp):
    return SensorMessage(lotId, spotId, timestamp, False)

parkingMessages = []
lots = [1, 2, 3]
timeObj = datetime.datetime.utcnow()

for i in range(0, 200000):
    lotId = random.choice(lots)
    spotId = random.randint(1, 100)
    startTime = timeObj.isoformat()
    endTime = (timeObj + datetime.timedelta(minutes=random.randint(10, 120))).isoformat()
    
    parkingMessages.append(startParkingSession(lotId, spotId, startTime))
    parkingMessages.append(endParkingSession(lotId, spotId, endTime))

parkingMessages.sort(key=lambda msg: msg.timeStamp)

for msg in parkingMessages:
    x = f"{msg.lotId}, {msg.spotId}, {msg.timeStamp}, {msg.occupied}"
    producer.send('smartparking', value=x.encode('utf-8'))

producer.flush()
