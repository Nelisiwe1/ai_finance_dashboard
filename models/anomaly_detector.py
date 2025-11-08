import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(data: pd.DataFrame):
    model = IsolationForest(contamination=0.1, random_state=42)
    data['anomaly'] = model.fit_predict(data[['Amount']])
    anomalies = data[data['anomaly'] == -1]
    return anomalies
