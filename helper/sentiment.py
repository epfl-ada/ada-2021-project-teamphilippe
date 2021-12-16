import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import json
import os
from tqdm import tqdm
from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 12, 9

def compute_sentiments(df_chunk, args):
  """
  Compute the polarity scores for each quotes in
  the dataframe.
  
  Parameters:
      df_chunk : chunk to process

  Returns:
    df : dataframe with positive (pos), negative (neg),
         neutral (neu), compound (compound) columns
  """
  def extract_scores(row):
    scores = analyzer.polarity_scores(row['quotation'])

    return pd.Series(scores)
  
  analyzer = SentimentIntensityAnalyzer()

  scores = df_chunk.apply(extract_scores, axis=1)
  print(scores.shape)
  
  # Add the new cols to the existing row
  return pd.concat([df_chunk, scores], axis=1), args

def avg_polarity_scores_by_period(df, period='%Y-%m'):
  """
  Compute the average negative and positive scores by period.
  
  Parameters:
      df : df to process
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%m-%V')

  Returns:
    df_scores: aggregated scores by period
  """
  return df.groupby(df.date.dt.strftime(period)).mean()[['neg', 'pos']]


def compute_category_prop_by_period(df, thresholds=(-0.33, 0.33), period='%Y-%m'):
  """
  Compute the proportions of sentiments by period (month or week).
  
  Parameters:
      df : df to process
      thresholds: limit compound score for each category
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%m-%V')

  Returns:
    category_by_period : df of the proportions of each category by period
  """
  # Compute how many quotes of each category there is each month
  compound_scores = df.apply(lambda row: -1 if row['compound'] <= thresholds[0] else 1 if row['compound'] >= thresholds[1] else 0, axis=1)
  category_by_period = compound_scores.groupby(df.date.dt.strftime(period)).value_counts()

  # Reset index to get the category counts as columns
  category_by_period = category_by_period.reset_index().rename(columns={0: 'count', 'level_1': 'category'})
  category_by_period = category_by_period.pivot('date', 'category', 'count')

  # Compute proportions by month of each category
  return category_by_period.div(category_by_period.sum(axis=1), axis=0)


def plot_sentiments_category_by_period(category_by_period):
  """
  Plot the evolutions of the proportions of sentiments by period.
  
  Parameters:
      df : df to process
  """
  ax = sns.lineplot(data=category_by_period)

  plt.title('Evolution of negative (-1), neutral (0), positive (1) sentiments')

  plt.xticks(rotation=90)
  plt.ylabel('Proportion')
  ax.xaxis.set_major_locator(plt.MaxNLocator(70))
  plt.show()


def compute_moving_average(df, n=6):
  """
  Compute the moving average of the given dataframe over
  the last n periods.
  
  Parameters:
      df : df to process
      n : number of periods to do the MA on

  Returns:
    df_ma : moving average over n periods
  """
  # Do a moving average considering the last n months
  return df.rolling(n, min_periods=1).mean()


def plot_means_sentiments_by_period(df, period='%Y-%m'):
  """
  Plot the evolutions of the mean polarity scores by period.
  
  Parameters:
      df : df to process
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%V')

  Returns:
    means_by_period : df of the mean polarity scores by period
  """
  # Group data by month and year
  means_by_period = df.groupby(df.date.dt.strftime(period))[['neg', 'neu', 'pos', 'compound']].mean()

  ax = sns.lineplot(data=means_by_period)
  plt.title('Evolution of the mean polarity scores by period')
  plt.xticks(rotation=90)
  plt.ylabel('Mean score over 1 period')
  ax.xaxis.set_major_locator(plt.MaxNLocator(70))
  plt.show()

  return means_by_period


def full_plots(file, period='%Y-%m'):
  """
  Simply call the above functions to do the full analysis.
  
  Parameters:
      file : file containing the polarity scores of the quotes
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%V')
  """
  # Load the file with the polarity scores
  df = pd.read_json(file, lines=True, compression='bz2')

  # Compute the proportions of each sentiments (negative, neutral, positive)
  # by period and plot it
  category_by_period = compute_category_prop_by_period(df, period=period)
  plot_sentiments_category_by_period(category_by_period)

  # Compute the mean of each polarity scores by month and plot it
  plot_means_sentiments_by_period(df)

def nb_quotes_by_period(df, period='%Y-%m'):
  """
  Count the number of quotes by period.
  
  Parameters:
      df : df to process
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%V')
  """
  # Count total number of quotes per period
  quotes_per_period = df.groupby(df.date.dt.strftime(period)).count()['quoteID']

  ax = sns.lineplot(x=quotes_per_period.index, y=quotes_per_period)

  ax.xaxis.set_major_locator(plt.MaxNLocator(70))
  plt.title('Number of quotes by period')
  plt.xticks(rotation=90)
  plt.ylabel('Count')
  plt.show()
