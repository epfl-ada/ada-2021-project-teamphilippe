import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def load_stock_csv(file, first_date=pd.to_datetime('2015-01-01', format='%Y-%m-%d'),
                   last_date=pd.to_datetime('2020-04-16', format='%Y-%m-%d')):
  """
  Load the csv file containing stock prices
  
  Parameters:
      file : file path of the csv
      first_date : first date of data to keep
      last_date : last date of data to keep

  Returns:
    df : df of the stock prices
  """
  df = pd.read_csv(file, index_col='Date', parse_dates=True)

  # Convert the prices cols to float (remove $ sign)
  prices_cols = ['Close/Last', 'Open', 'High', 'Low']
  df[prices_cols] = df.apply(lambda row: pd.Series([float(row[col][1:]) for col in prices_cols]), axis=1)

  # Only keeps data between first date and last date both included
  df = df[(df.index >= first_date) & (df.index <= last_date)]

  return df


def aggregate_stock_by_period(df, period='%Y-%m'):
  """
  Aggregate the daily values of the stock prices by period by
  computing the mean of each column as well as the mean of the daily diff as
  well as the avg between the open and close prices.
  
  Parameters:
      df : df of the daily stock prices
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%m-%V')
      
  Returns:
      df : df of the aggregated stock prices
  """
  # Compute mean between open and close for each day
  df['diff_op_cl'] = (df['Close/Last'] - df['Open'])
  df['avg_op_cl'] = (df['Close/Last'] + df['Open']) / 2

  # Compute the mean by month for all columns
  return df.groupby(df.index.strftime(period)).mean()


def plot_stocks_by_period(df, stock, filename=''):
  """
  Plot the avg open/close price by period.

  Parameters:
      df : df of the aggregated stock prices
      stock : 
      filename : The file name under which we should save the figure
  """
  fig = go.Figure()

  text = []
  index = []
  for date, val in zip(df.index, df['avg_op_cl'].to_list()):
    year, week = date.split('-')
    text.append(f'{(round(val, 4))}<br>Date:{week} week of {year}')
    index.append(f'{week} week of {year}')

  fig.add_trace(go.Scatter(x=index, y=df['avg_op_cl'],
                            text=text,
                            hoverinfo='text'
                          ))
  
  fig.update_layout(title={
        'text': f'Evolution of stock prices of {stock}',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title='Avg between open and close prices [$]',
        xaxis_title='Date')
  fig.show()
  if filename!='':
    fig.write_html(filename)


def cross_correlation(ts_x, ts_y, lag=0):
  """
  Compute the lagged cross-correlation between 2 time series.
  
  Parameters:
      ts_x: first time series
      ts_y: second time series
      lag: number of months to shift
    Returns:
      float: cross correlation
  """
  return ts_x.corr(ts_y.shift(lag))


def all_cross_correlations(ts_x, ts_y, lags=range(9)):
  """
  Compute all the lagged cross-correlation between 2 time series
  with the given lags.
  
  Parameters:
      ts_x: first time series
      ts_y: second time series
      lags: lags to consider
  Returns:
      list: all the lagged cross correlations
  """
  return [cross_correlation(ts_x, ts_y, i) for i in lags]


def plot_cross_correlations(cc, x):
  """
  Plot the lagged cross correlations.
  
  Parameters:
      cc: lagged correlations
      x: lags in months
  """
  sns.lineplot(x=x, y=cc)
  plt.title('Cross correlations')
  plt.xlabel('Lag [months]')
  plt.ylabel('Correlation')
  plt.show()