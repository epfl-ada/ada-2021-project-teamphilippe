import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import itertools
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.patches as mpatches
from group import *
import plotly.express as px
import plotly.graph_objects as go
from sentiment import *

def full_plots_visualisations(file, sentiments_category_filename="", means_sentiments_filename="", period='%Y-%m'):
  """
  Simply call the above functions to do the full analysis.
  
  Parameters:
      file : file containing the polarity scores of the quotes
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%m-%V')
  """
  # Load the file with the polarity scores
  df = pd.read_json(file, lines=True, compression='bz2')

  # Compute the proportions of each sentiments (negative, neutral, positive)
  # by period and plot it
  category_by_period = compute_category_prop_by_period(df, period=period)
  plot_sentiments_category_by_period_visualisations(category_by_period, filename=sentiments_category_filename)

  # Compute the mean of each polarity scores by month and plot it
  plot_means_sentiments_by_period_visualisations(df, filename=means_sentiments_filename)

def plot_sentiments_category_by_period_visualisations(category_by_period, filename=""):
  """
  Plot the evolutions of the proportions of sentiments by period.
  
  Parameters:
      df : df to process
      filename: The file name under which we should save the figure
  """
  name = {-1:'negative', 0:'neutral', 1:'positive'}
  fig = go.Figure()
  for col in [0, -1, 1]:
    cur_data = category_by_period[col]
    text = []
    for date, val in zip(cur_data.index, cur_data.to_list()):
      text.append(f'{(round(val, 4))}<br>Date:{date}<br>Name:{name[col]}')
    fig.add_trace(go.Scatter(x=cur_data.index, y=cur_data,
                              name=name[col],
                              text=text,
                              hoverinfo='text'
                             ))
  fig.update_layout(title={
        'text': f'Proportion of sentiments over the years',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        legend_title="Sentiment",
        yaxis_title="Proportion",
        xaxis_title="Date")
  fig.show()
  if filename!="":
    fig.write_html(filename)

def plot_means_sentiments_by_period_visualisations(df, period='%Y-%m', filename=""):
  """
  Plot the evolutions of the mean polarity scores by period.
  
  Parameters:
      df : df to process
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%V')
      filename: The file name under which we should save the figure
  """
  # Group data by month and year
  means_by_period = df.groupby(df.date.dt.strftime(period))[['neg', 'neu', 'pos', 'compound']].mean()

  name = {'neg':'negative', 'neu':'neutral', 'pos':'positive', 'compound':'compound'}
  fig = go.Figure()
  for col in ['neu', 'neg', 'pos', 'compound']:
    cur_data = means_by_period[col]
    text = []
    index = []
    for date, val in zip(cur_data.index, cur_data.to_list()):
      if(period == '%Y-%V'):
        year, week = date.split('-')
        text.append(f'{(round(val, 4))}<br>Date:{week} week of {year}<br>Name:{name[col]}')
        index.append(f'{week} week of {year}')
      else:
        text.append(f'{(round(val, 4))}<br>Date:{date}<br>Name:{name[col]}')
        index.append(date)
      
    fig.add_trace(go.Scatter(x=index, y=cur_data,
                              name=name[col],
                              text=text,
                              hoverinfo='text'
                            ))
  fig.update_layout(title={
        'text': 'Evolution of the mean polarity scores over time',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        legend_title="Polarity score",
        yaxis_title="Mean score",
        xaxis_title="Date")
  fig.show()
  if filename!="":
    fig.write_html(filename)

def nb_quotes_by_period_visualisations(df, period='%Y-%m', filename=""):
  """
  Count the number of quotes by period.
  
  Parameters:
      df : df to process
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%V')
      filename: The file name under which we should save the figure
  """
  # Count total number of quotes per period
  quotes_per_period = df.groupby(df.date.dt.strftime(period)).count()['quoteID']

  fig = go.Figure()

  text = []
  index = []
  for date, val in zip(quotes_per_period.index, quotes_per_period.to_list()):
    if(period == '%Y-%V'):
      year, week = date.split('-')
      text.append(f'{(round(val, 4))}<br>Date:{week} week of {year}')
      index.append(f'{week} week of {year}')
    else:
      text.append(f'{(round(val, 4))}<br>Date:{date}')
      index.append(date)
  fig.add_trace(go.Scatter(x=index, y=quotes_per_period,
                            text=text,
                            hoverinfo='text'
                          ))
  
  fig.update_layout(title={
        'text': 'Number of quotes by period',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title="Count",
        xaxis_title="Date")
  fig.show()
  if filename!="":
    fig.write_html(filename)
