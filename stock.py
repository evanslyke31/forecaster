from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

def accuracy_score(observed,true):
    diff = 0.0
    for a,b in zip(observed,true):
        diff += abs((float(a) - b)/b)
    return 1.0 - diff

#dataset = read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=AAPL&interval=5min&apikey=F4J8FATPAPG9HGF3&datatype=csv")
symbols = read_csv("constituents.csv")


for sym in np.array(symbols['Symbol']):
    data = read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=%s&interval=5min&apikey=F4J8FATPAPG9HGF3&datatype=csv" % (sym))
    print np.array(data['close'])
"""forcast_weight = 100

array = dataset.values
Y = array[:,4]
X = np.arange(len(array[:,0]))
Y_train = Y[:len(Y)-forcast_weight]
Y_validation = Y[len(Y)-forcast_weight:]

model = ARIMA(Y_train, order=(0, 1, 1))
fit = model.fit(disp=False)
predictions = fit.predict(len(Y)-forcast_weight,len(Y),typ='levels')
predictions = np.array(predictions)
#print predictions
#print Y_validation
print accuracy_score(predictions, Y_validation)

fig = pyplot.figure()
ax = pyplot.axes()
ax.plot(range(len(Y)),Y,color='red')
ax.plot(range(len(Y) - len(predictions),len(Y)),predictions,color='blue')
#pyplot.style.use('seaborn-whitegrid')
#pyplot.plot(Y, color='red')
#pyplot.plot(predictions, color='blue')
pyplot.show()
"""