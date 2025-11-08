import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def predict_future_spending(data: pd.DataFrame, days_ahead=30):
    data['Date'] = pd.to_datetime(data['Date'])
    data['DateOrdinal'] = data['Date'].map(pd.Timestamp.toordinal)

    X = data[['DateOrdinal']]
    y = data['Amount']

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(data['Date'].max(), periods=days_ahead)
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_ordinals)

    return pd.DataFrame({'Date': future_dates, 'Predicted_Amount': predictions})
