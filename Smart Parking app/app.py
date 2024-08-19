
from flask import Flask, render_template
from cassandra.cluster import Cluster

# Initialize Flask
app = Flask(__name__)

# Initialize Cassandra Cluster and Session
cluster = Cluster()
session = cluster.connect('test')

@app.route('/')
def index():
    lot1 = session.execute('SELECT occrate, price FROM smartpark WHERE lotid = 1 LIMIT 1').one()
    lot2 = session.execute('SELECT occrate, price FROM smartpark WHERE lotid = 2 LIMIT 1').one()
    lot3 = session.execute('SELECT occrate, price FROM smartpark WHERE lotid = 3 LIMIT 1').one()
    
    price1 = str(lot1.price) if lot1 else 'N/A'
    price2 = str(lot2.price) if lot2 else 'N/A'
    price3 = str(lot3.price) if lot3 else 'N/A'
    
    occrate1 = str(lot1.occrate) if lot1 else 'N/A'
    occrate2 = str(lot2.occrate) if lot2 else 'N/A'
    occrate3 = str(lot3.occrate) if lot3 else 'N/A'
    
    return render_template('index.html', 
                           price1=price1, price2=price2, price3=price3,
                           occrate1=occrate1, occrate2=occrate2, occrate3=occrate3)

@app.route('/ParkingLot/<int:pid>')
def ParkingLot(pid=1):
    lot = session.execute('SELECT occrate, price FROM smartpark WHERE lotid = %s LIMIT 30', (pid,))
    lot_data = list(lot)
    
    if lot_data:
        price = lot_data[0].price
        occrate = lot_data[0].occrate
    else:
        price = 'N/A'
        occrate = 'N/A'
    
    return render_template('plot.html', pid=pid, price=price, occrate=occrate, lot=lot_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
