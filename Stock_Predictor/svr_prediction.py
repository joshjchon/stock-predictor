import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR



def read_csv_data_final(filename):
    days, valuations = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for line in reader:
            days.append(int(line[0].split('/')[1]))
            valuations.append(float(line[1].replace('$', '')))
    return days, valuations

def model_prediction(day_list, valuation_list, target_day):
    day_list = np.reshape(day_list, (len(day_list), 1))
    model_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model_linear = SVR(kernel='linear', C=1e3)
    model_poly = SVR(kernel='poly', C=1e3, degree=2)
    
    model_rbf.fit(day_list, valuation_list)
    model_linear.fit(day_list, valuation_list)
    model_poly.fit(day_list, valuation_list)
    
    plt.scatter(day_list, valuation_list, color='black', label='Data')
    plt.plot(day_list, model_rbf.predict(day_list), color='red', label='RBF Model')
    plt.plot(day_list, model_linear.predict(day_list), color='green', label='Linear Model')
    plt.plot(day_list, model_poly.predict(day_list), color='blue', label='Polynomial Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return model_rbf.predict([[target_day]])[0], model_linear.predict([[target_day]])[0], model_poly.predict([[target_day]])[0]

if __name__ == "__main__":
    days, valuations = read_csv_data_final('HistoricalData.csv')
    forecasted_valuation = model_prediction(days, valuations, 29)
    print(forecasted_valuation)
