from flask import Flask, render_template, jsonify
import pandas as pd
from models.anomaly_detector import detect_anomalies
from models.predictor import predict_future_spending

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/transactions')
def transactions():
    data = pd.read_csv('data/transactions.csv')
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/anomalies')
def anomalies():
    data = pd.read_csv('data/transactions.csv')
    anomalies = detect_anomalies(data)
    return jsonify(anomalies.to_dict(orient='records'))

@app.route('/api/predictions')
def predictions():
    data = pd.read_csv('data/transactions.csv')
    preds = predict_future_spending(data)
    return jsonify(preds.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
