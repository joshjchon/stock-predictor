
# Stock Price Predictor Using SVR

## Description
This project predicts stock prices using Support Vector Regression (SVR). The prediction is based on historical stock price data. Three different SVR models, each with a different kernel (RBF, Linear, and Polynomial), are trained on the data. The models' predictions are then visualized alongside the actual stock prices to evaluate their performance. Made by Anirudh Nagasamudra and Joshua Chon.

## Installation
Ensure you have the following Python libraries installed:
- numpy
- matplotlib
- scikit-learn

You can install them using pip:
```
pip install numpy matplotlib scikit-learn
```

## Usage
1. Place your stock price data in a CSV file. The CSV should have dates in the "MM/DD/YYYY" format and prices with a preceding dollar sign, like "$123.45".
2. Update the script `svr_prediction.py` with the path to your CSV file.
3. Run the script:
```
python3 svr_prediction.py
```
4. The script will display a chart showing the actual stock prices and the predictions made by the three SVR models. It will also print the predicted prices for a specified date.

## Results
The chart produced by the script represents the stock prices over a series of days, as well as the predictions made by the three SVR models. The black dots represent actual stock prices. The red, green, and blue lines represent predictions from the RBF, Linear, and Polynomial SVR models, respectively.

By comparing the prediction lines to the actual data points, you can gauge how well each model fits the historical stock price data.

## Contributing
Feel free to fork the repository and submit pull requests.

## License
This project is open source, under the MIT License.

