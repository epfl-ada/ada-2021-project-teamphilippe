import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

plt_aspect=2
plt.figure(figsize=(12,9))

def describe(df, year):
  """
    describe the distribution of the data frame

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  print(year, ':')
  print(df['numOccurrences'].describe())

def num_occurences(df, year):
  """
    plot the distribution of the number of occurences per quotes

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  ax = sns.displot(data=df, x='numOccurrences', log_scale=(True, True), aspect=plt_aspect)
  plt.suptitle(f'Histogram of number of quotes per number of occurences ({year})')
  ax.set(xlabel="Number of occurences",ylabel="Number of quotes")
  plt.show()

def set_count(df, year, attribut):
  """
    count occurence of each element

    Parameters:
      df :        the data frame of a specific year
      year :      the year
      attribut :  attribut
      
    Returns:
      count : set of key to number of occurence
      newdf : the data frame to 
  """
  count = {}
  key = f'speaker_{attribut}'
  def update(row):
    if(row != None):
      for element in row:
        if(element in count):
          count[element] += 1
        else :
          count[element] = 1
  df[key].apply(update)
  return count

def set_attribut(df, year, attribut):
  """
    plot a distribution of the number of quotes per `attribut`

    Parameters:
      df :        the data frame of a specific year
      year :      the year
      attribut :  attribut
  """
  # Count occurence of each element 
  count = set_count(df, year, attribut)

  # Plot the number of quotes per attribut histogram
  attribut = attribut.replace('_', ' ')
  attribut = attribut[:1].upper() + attribut[1:]
  ax = sns.displot(data = count, log_scale = (True, True), bins = 100, aspect=plt_aspect)
  ax.set(xlabel="Number of quotes", ylabel=f'Number of {attribut}')
  plt.suptitle(f'Number of {attribut} by number of Quotes ({year})')
  plt.show()

def set_specific_attribut(df, year, attribut, rotation=0):
  """
    plot a distribution of the number of quotes per `attribut`

    Parameters:
      df :        the data frame of a specific year
      year :      the year
      attribut :  attribut
  """
  # Count occurence of each element 
  dicti = set_count(df, year, attribut)

  # Plot the number of quotes per attribut histogram
  attribut = attribut.replace('_', ' ')
  attribut = attribut[:1].upper() + attribut[1:]
  newdf = pd.DataFrame(list(dicti.items()), columns =[attribut,"Count"]).sort_values(['Count'], ascending=False).reset_index(drop=True)

  plt.figure(figsize=(12,9))
  ax = sns.barplot(x=newdf.index, y=newdf.Count, log=(False, True), color='C0')
  ax.set_yscale("log")
  ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
  ax.set(xlabel=attribut, ylabel='Count')
  plt.suptitle(f'Number of {attribut} by number of Quotes ({year})')

  # add proper Dim values as x labels
  ax.set_xticklabels(newdf[attribut])
  for item in ax.get_xticklabels():
    item.set_rotation(rotation)
  for i, v in enumerate(newdf["Count"].iteritems()):
    text = "{:,}".format(v[1])
    ax.text(i , v[1], text, color='black', va='bottom', ha='center', rotation=0)
  plt.show()

def set_to_string(set_):
  """
    return the set in string

    Parameters:
      set_ :   set

    Returns:
      string : string
  """
  if set_ == None :
    return "None"
  s = ""
  for v in set(set_) :
    s += ", " + v
  return s[2:]

def to_set(x):
  """
    return the set of the list

    Parameters:
      x :   list

    Returns:
      set : set
  """
  if x == None :
    return None
  return set(x)

def quotes_by_author(df, year):
  """
    plot the distribution of the number of quotes by author

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  # Count the number of quote per author
  df['Author'] = df.apply(lambda row : row['qid'], axis=1)
  var = df.groupby('Author').agg(count=pd.NamedAgg(column="Author", aggfunc="count"))
  # Display the distribution
  ax =sns.displot(data=var, x='count', log_scale=(True, True), aspect=plt_aspect)
  ax.set(xlabel="Quotes count per author",ylabel="Number of author")
  plt.suptitle(f'Histogram of number of quotes by Author ({year})')
  plt.show()

def quotes_per_ages(df, year):
  """
    plot a histogram of the number of quotes by ages

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  df = df[0 < df['speaker_age']]
  df = df[df['speaker_age'] <= 120]
  # Create a histogram of the number of quotes per age
  ax = sns.displot(data=df[pd.notna(df['speaker_age'])], x='speaker_age', log_scale=(False, True), bins=120, aspect=plt_aspect)
  ax.set(xlabel="Ages",ylabel="Count per ages")
  plt.suptitle(f'Number of quotes per ages ({year})')
  plt.show()

def quotes_per_sex(df, year):
  """
    plot a distribution of the number of quotes per sex

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_specific_attribut(df, year, 'gender', 90)

def quotes_per_continent(df, year):
  """
    plot a distribution of the number of quotes per continent

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_specific_attribut(df, year, 'continent')

def quotes_per_nationality(df, year):
  """
    plot a distribution of the number of quotes per nationality

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_attribut(df, year, 'nationality')

def quotes_per_ethnic_group(df, year):
  """
    plot a distribution of the number of quotes per ethnic group

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_attribut(df, year, 'ethnic_group')

def quotes_per_occupation(df, year):
  """
    plot a distribution of the number of quotes per occupation

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_attribut(df, year, 'occupation')

def quotes_per_party(df, year):
  """
    plot a distribution of the number of quotes per party

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_attribut(df, year, 'party')

def quotes_per_academic_degree(df, year):
  """
    plot a distribution of the number of quotes per academic degree

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_specific_attribut(df, year, 'academic_degree_group')

def quotes_per_candidacy(df, year):
  """
    plot a distribution of the number of quotes per candidacy

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_attribut(df, year, 'candidacy')

def quotes_per_religion(df, year):
  """
    plot a distribution of the number of quotes per religion

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  set_attribut(df, year, 'religion')

def quotes_per_day(df, year):
  """
    plot a distribution of the number of quotes per day of year

    Parameters:
      df :    the data frame of a specific year
      year :  the year
  """
  # Compute the number of quotes per day of year
  df['day_of_year'] = df['date'].apply(lambda x: x.day_of_year)
  day_of_month = df[df['date'].apply(lambda x: int(str(x)[8:10]) != 1)]

  # Plot the line plot for the full year
  from matplotlib import rcParams
  rcParams['figure.figsize'] = 25,4
  sns.histplot(data=df, x='day_of_year', binwidth=1, color='r')
  sns.histplot(data=day_of_month, x='day_of_year', binwidth=1)
  ax = sns.histplot(data=day_of_month, x='day_of_year', binwidth=1)
  ax.set(xlabel="Day of year",ylabel="Count per day")
  plt.suptitle(f'Number of quotes per day ({year})')
  plt.show()

def describe_year(df, year):
  print(f"==== Exploration sampled quotes for year {year} ====")
  # Statistical description of this year data.
  print("\n\nStatistical summary\n\n")
  print(df.describe())
  # Display all the plot for the current year
  print("\n\nVisualisations\n\n")
  numOccurences(df, year)
  quotesByAuthor(df, year)
  quotesPerAges(df[df['speaker_age'] < 120], year)
  quotesPerSex(df, year)
  quotesPerNationality(df, year)
  quotes_per_day(df, year)


def plot_all_years(func, df_zip):
  for df, year in df_zip:
    func(df, year)
    print()