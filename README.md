!pip install prophet plotly openpyxl

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
# Load CSV file
df = pd.read_csv('/content/covid_19_clean_complete.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Preview
df.head()

global_data = df.groupby('Date').sum().reset_index()

global_data.head()

global_data['Active'] = (
    global_data['Confirmed']
    - global_data['Deaths']
    - global_data['Recovered']
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=global_data['Date'],
    y=global_data['Confirmed'],
    mode='lines',
    name='Confirmed'
))

fig.add_trace(go.Scatter(
    x=global_data['Date'],
    y=global_data['Recovered'],
    mode='lines',
    name='Recovered'
))

fig.add_trace(go.Scatter(
    x=global_data['Date'],
    y=global_data['Deaths'],
    mode='lines',
    name='Deaths'
))

fig.update_layout(
    title="Global COVID-19 Trend",
    xaxis_title="Date",
    yaxis_title="Number of Cases"
)

fig.show()
global_data['Recovery Rate'] = global_data['Recovered'] / global_data['Confirmed']
global_data['Death Rate'] = global_data['Deaths'] / global_data['Confirmed']

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=global_data['Date'],
    y=global_data['Recovery Rate'],
    mode='lines',
    name='Recovery Rate'
))

fig.add_trace(go.Scatter(
    x=global_data['Date'],
    y=global_data['Death Rate'],
    mode='lines',
    name='Death Rate'
))

fig.update_layout(
    title="Recovery and Death Rate Over Time",
    xaxis_title="Date",
    yaxis_title="Rate"
)

fig.show()

prophet_df = global_data[['Date', 'Confirmed']]
prophet_df.columns = ['ds', 'y']

prophet_df.head()


model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

forecast.tail()

fig = plot_plotly(model, forecast)
fig.update_layout(title="COVID-19 Confirmed Cases Prediction (Next 7 Days)")
fig.show()

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

india_data = df[df['Country/Region'] == 'India']
india_data = india_data.groupby('Date').sum().reset_index()

fig = px.line(india_data, x='Date', y='Confirmed',
              title='India COVID-19 Confirmed Cases Trend')
fig.show()
