import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from group import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def cumsum_prop(group):
    """
    Get the cumulative sum in the following order: Positive, neutral, negative
    Parameters:
      - group: The group in which we should compute the cumulative sum
    Returns:
      - Return the group with the cumulative sum
    """
    group['prop'] = group['prop'].cumsum()
    return group


def plot_proportion_over_ages_plotly(df, col, list_values, legend_title="",belonging_fct=lambda x, lst: lst is not None and x in lst,
                                     investigated_person="Mark Zuckerberg", investigated_attributes="continent", filename=""):
    """
    Plot the proportion of author by the give
    Parameters:
      - df: the dataframe from which we need to compute the proportion
      - col: The column from which we need to compute the proportion
      - list_values: the list of values we want to investigate
      - belonging_fct: The function we need to use to test the belonging of someone to a specific group
      - investigated_person: The person we are currently investigating
      - investigated_attributes: The attributes we are investigating
      - filename: The file name under which we should save the figure
    """

    def format_text(row, val):
        return f'{val}<br>Date:{(row["date"])}<br>Proportion:{row["prop"]:.4e}<br>Count:{(row["count"])}'

    df2 = df.copy()
    df2['cat'] = "Other"
    values_copy = list_values.copy()
    for val in list_values:
        mask = df2.apply(lambda x: belonging_fct(val, x[col]), axis=1)
        df2.loc[mask, 'cat'] = str(val)
    # Add the other category to the plot.
    values_copy.append("Other")

    # Compute how many quotes of each category there is each month
    category_by_month = df2[['date', 'cat']].groupby([df2.date.dt.strftime('%Y-%m'), 'cat']).agg(
        count=pd.NamedAgg(column='cat', aggfunc="count"))

    # Reset index to get the category counts as columns
    category_by_month = category_by_month.reset_index()

    # Compute proportions by month of each category
    sum_by_cat = \
        df2.groupby(df2.date.dt.strftime('%Y-%m')).agg(count=pd.NamedAgg(column='cat', aggfunc="count")).to_dict()[
            'count']
    category_by_month['prop'] = category_by_month.apply(lambda x: x['count'] / sum_by_cat[x['date']], axis=1)

    fig = go.Figure()
    # figure size in inches
    for i, name in enumerate(list_values):
        cur_data = category_by_month[category_by_month['cat'] == str(name)]
        fig.add_trace(go.Scatter(x=cur_data['date'], y=cur_data['prop'],
                                 fillcolor=px.colors.qualitative.Set3[i + 2], name=str(name),
                                 text=cur_data.apply(lambda x: format_text(x, name), axis=1).to_list(),
                                 hoverinfo='text'))

    fig.update_layout(title={
        'text': f'Proportion of quotes talking about {investigated_person} by {investigated_attributes}',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        legend_title =legend_title,
        yaxis_title="Proportion of the quotes",
        xaxis_title="Date")
    fig.show()
    
    # Save plot
    if(filename!=''):
        fig.write_html(filename)


def plot_mean_sentiment_by_attr_value(df, col, col_title, list_values, legend_title ="",
                                      belonging_fct=lambda val, lst: lst is not None and val in lst,
                                      investigated_person="Mark Zuckerberg", use_plotly=False,filename=""):
    """
    Plot the proportion of each sentiment in a given range
    Parameters:
      - df : The dataframe containing all the quotes
      - cat: the column we are investigating
      - val: The current value we are looking for
      - investigated_person: The person we are currently investigating
      - use_plotly: If we need to use plotly instead of seaborn
      - filename: The file name under which we should save the figure
  """
    df['assigned_value'] = 'Other'
    for index, val in enumerate(list_values):
        mask = df.apply(lambda row: belonging_fct(val, row[col]), axis=1)
        df.loc[mask, "assigned_value"] = index
    df = df[df['assigned_value'] != 'Other']

    mean_compound_month = df.groupby([df.date.dt.strftime('%Y-%m'), 'assigned_value']).agg(
        mean=pd.NamedAgg(column='compound', aggfunc="mean"), count=pd.NamedAgg(column='compound', aggfunc="count"))

    # Reset index to get the category counts as columns
    mean_compound_month = mean_compound_month.reset_index()

    # figure size in inches
    # Create the line plot
    if use_plotly:
        def format_text(row, val):
            return f'{val}<br>Date:{(row["date"])}<br>Mean:{row["mean"]:.4e}<br>Count:{(row["count"])}'

        fig = go.Figure()
        for i, val in enumerate(list_values):
            cur_data=mean_compound_month[mean_compound_month['assigned_value'] == i]
            fig.add_trace(
                go.Scatter(x=cur_data['date'],
                           y=cur_data['mean'],
                           marker=dict(color=px.colors.qualitative.Set3[i + 2]),
                           fillcolor=px.colors.qualitative.Set3[i + 2], name=f'{val}',
                           text=cur_data.apply(
                               lambda x: format_text(x, val), axis=1).to_list(),
                           hoverinfo='text'))
        fig.update_layout(title={
            'text': f'Mean compound score of quotes talking about {investigated_person} by {col_title}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            legend_title =legend_title,
            yaxis_title="Mean compound score",
            xaxis_title="Date")
        fig.show()

        # Save plot
        if(filename!=''):
            fig.write_html(filename)
    else:
        rcParams['figure.figsize'] = 12, 8
        ax = sns.lineplot(data=mean_compound_month, x='date', y='mean', hue='assigned_value',
                          palette=sns.color_palette()[0:len(list_values)])
        plt.title(f'Proportion of the sentiment in the quotes talking about {investigated_person} by {col_title}')
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, list_values, title=col_title)
        plt.xticks(rotation=90)
        plt.xlabel("Time")
        plt.ylabel('Mean compound score')
        plt.show()


def plot_proportion_over_year(df, col, investigated_person="Mark Zuckerberg", investigated_attributes="ages"):
    """
    Plot the proportion of author by the give
    Parameters:
        - df: the dataframe from which we need to compute the proportion
        - col: The column from which we need to compute the proportion
        - investigated_person: The person we are currently investigating
        - investigated_attributes: The attributes we are investigating
    """
    # Compute how many quotes of each category there is each month
    category_by_month = df[['date', col]].groupby([df.date.dt.strftime('%Y-%m'), col]).agg(
        count=pd.NamedAgg(column=col, aggfunc="count"))

    # Reset index to get the category counts as columns
    category_by_month = category_by_month.reset_index()
    category_by_month = category_by_month.pivot('date', col, 'count')

    # Compute proportions by month of each category
    category_by_month = category_by_month.div(category_by_month.sum(axis=1), axis=0)

    # figure size in inches
    rcParams['figure.figsize'] = 15, 10
    sns.lineplot(data=category_by_month)
    plt.title(f'Proportion of quotes talking about {investigated_person} by {investigated_attributes}')
    plt.xticks(rotation=90)
    plt.ylabel('Proportion of the quotes')
    plt.show()


def plot_proportion_by_ages_over_year(df, tuple_list):
    """
    Plot the proportion of quotes by age range over the years
    Parameters:
        - df : The dataframe containing all the quotes
        - tuple_list: A list of tuple containing tuples of upper and lower limit
    """
    # Add the age range for each rowin the dataframe
    df2 = df.copy()
    df2['age_cat'] = "Other"
    for (lim_inf, lim_sup) in tuple_list:
        df2['age_cat'] = df2.apply(
            lambda x: f"{lim_inf}-{lim_sup}" if lim_inf <= x['speaker_age'] <= lim_sup else x['age_cat'], axis=1)

    # Compute the number of person in each age range
    for (lim_inf, lim_sup) in tuple_list:
        age_cat = f"{lim_inf}-{lim_sup}"
        print(f"Number of person in the range {age_cat} is {(df2['age_cat'] == age_cat).sum()}.")
    # Plot the proportion over the ages
    plot_proportion_over_year(df2, "age_cat")


def plot_sentiment_for_range(df, cat, val, investigated_person="Mark Zuckerberg"):
    """
    Plot the proportion of each sentiment in a given range
    Parameters:
        - df : The dataframe containing all the quotes
        - cat: the column we are investigating
        - val: The current value we are looking for
        - investigated_person: The person we are currently investigating
    """
    mask = df[cat] == val
    df = df[mask]

    category_by_month = df[['date', cat, 'sentiment']].groupby([df.date.dt.strftime('%Y-%m'), 'sentiment']).agg(
        count=pd.NamedAgg(column=cat, aggfunc="count"))

    # Reset index to get the category counts as columns
    category_by_month = category_by_month.reset_index()
    category_by_month = category_by_month.pivot('date', 'sentiment', 'count')
    # Compute proportions by month of each category
    category_by_month = category_by_month.div(category_by_month.sum(axis=1), axis=0)
    category_by_month = category_by_month.fillna(0)

    # Figure size in inches
    rcParams['figure.figsize'] = 12, 8
    sns.lineplot(data=category_by_month)
    plt.title(f'Proportion of the sentiment in the quotes talking about {investigated_person} by ages')
    plt.xticks(rotation=90)
    plt.ylabel('Proportion')
    plt.show()


def plot_proportion_sentiment_over_years(df, tuple_list):
    """
    Plot the proportion of sentiment in the quotes by age range over the years
    Parameters:
        - df : The dataframe containing all the quotes
        - tuple_list: A list of tuple containing tuples of upper and lower limit
    """
    df2 = df.copy()
    df2['age_cat'] = "Other"
    for (lim_inf, lim_sup) in tuple_list:
        df2['age_cat'] = df2.apply(
            lambda x: f"{lim_inf}-{lim_sup}" if lim_inf <= x['speaker_age'] <= lim_sup else x[
                'age_cat'], axis=1)
    for (lim_inf, lim_sup) in tuple_list:
        plot_sentiment_for_range(df2, "age_cat", f"{lim_inf}-{lim_sup}")


def plot_proportion_by_gender_over_year(df):
    """
    Plot the proportion of sentiment in the quotes by gender over the years
    Parameters:
        - df : The dataframe containing all the quotes
    """
    df2 = df.copy()
    df2['cat_sex'] = df2.apply(
        lambda x: "male" if x['speaker_gender'] is not None and "male" in x['speaker_gender'] else (
            "female" if x['speaker_gender'] is not None and "female" in x['speaker_gender'] else "Other"), axis=1)
    plot_proportion_over_year(df2, "cat_sex")


def plot_proportion_by_continent_over_year(df, list_values):
    """
    Plot the proportion of sentiment in the quotes by gender over the years
    Parameters:
        - df : The dataframe containing all the quotes
        - list_values: 
    """
    df2 = df.copy()
    df2['cat_continent'] = "Other"
    for val in list_values:
        mask = df2.apply(lambda x: x['speaker_continent'] is not None and val in x['speaker_continent'], axis=1)
        df2.loc[mask, 'cat_continent'] = val
    plot_proportion_over_year(df2, "cat_continent")


def boxplot_cluster(orig_df):
    """
    Plot a box plot for the mean compound score of each cluster
    Parameters:
        orig_df: The dataframe containing all the quotes
    """
    # Print a box plot for the compound score in each cluster
    plt.subplots(1, figsize=(10, 5))
    ax = sns.boxplot(x="assigned_cluster", y="compound", data=orig_df)
    plt.suptitle("Mean compound score by cluster")
    ax.set_xlabel("Assigned cluster")
    ax.set_ylabel("Mean compound score")
    plt.show()


def bar_plot_proportion(df_score, filename=""):
  """
  Plot a bar plot with 3 different bar for each cluster representing the proportion of each sentiment in each cluster
  Parameters:
      - df_score: The dataframe containing the proportion for each sentiment in each cluster.
      - filename: The file name under which we should save the figure
  """
  # Compute the distribution by sentiment type
  pos = []
  neu = []
  neg = []
  x = []
  def f(row):
    if row['sentiment'] > 0:
      pos.append(row['prop'])
    if row['sentiment'] == 0:
      neu.append(row['prop'])
    if row['sentiment'] < 0:
      neg.append(row['prop'])
    if row['assigned_cluster'] not in x:
      x.append(row['assigned_cluster'])

  df_score.sort_values('sentiment', ascending=False).apply(f, axis=1)

  # Bar plot with one bar for each sentiment proportion in each cluster
  fig = go.Figure(data=[
    go.Bar(name='Positive', x=x, y=pos),
    go.Bar(name='Neutral', x=x, y=neu),
    go.Bar(name='Negative', x=x, y=neg)
  ])
  fig.update_layout(title={
      'text': 'Proportion of each sentiment in each cluster',
      'y': 0.9,
      'x': 0.5,
      'xanchor': 'center',
      'yanchor': 'top'},
      legend_title="Sentiment",
      yaxis_title="Proportion of the cluster",
      xaxis_title="Assigned cluster")
  fig.show()

  # Save plot
  if(filename!=''):
    fig.write_html(filename)


def bar_plot_cum_sum(df_score, k):
    """
    Plot a bar plot with superposed positive, neutral and negative proportion of quotes.
    Parameters :
        df_score: The dataframe containing the proportion for each sentiment in each cluster
        k: The number of clusters
    """
    # Compute the histogram of the superposed proportion of sentiment
    cumsum_grouped = df_score.sort_values(['sentiment', 'prop'], ascending=False).groupby(
        'assigned_cluster').apply(cumsum_prop)

    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.set_xticks(list(range(k)))
    sns.barplot(x="assigned_cluster", y="prop", data=cumsum_grouped[cumsum_grouped['sentiment'] == -1],
                color='lightgrey', ax=ax)
    sns.barplot(x="assigned_cluster", y="prop", data=cumsum_grouped[cumsum_grouped['sentiment'] == 0],
                color='orange', ax=ax)
    sns.barplot(x="assigned_cluster", y="prop", data=cumsum_grouped[cumsum_grouped['sentiment'] == 1],
                color='royalblue', ax=ax)
    plt.suptitle("Proportion of each sentiment in each cluster")
    ax.set_xlabel("Assigned cluster")
    ax.set_ylabel("Proportion of the cluster")

    pos_patch = mpatches.Patch(color='royalblue', label='Positive')
    neu_patch = mpatches.Patch(color='orange', label='Neutral')
    neg_patch = mpatches.Patch(color='lightgrey', label='Negative')
    ax.legend(handles=[pos_patch, neu_patch, neg_patch], title="Sentiment", bbox_to_anchor=(1, 1))

    plt.show()


def plot_box_plot_mean_compound(df, col, list_values, col_name,
                                belonging_fct=lambda x, lst: (lst is not None) and (x in lst), investigated_person="Mark Zuckerberg",filename='box_plot.html'):
    """
      Plot a box plot for the mean compound score for each value of the given list of values.

      Parameters:
        - df: the dataframe containing the score
        - col: the column on which we should group the quotes
        - list_values: the list of values on which we want to group the quotes
        - legend_prefix: The prefix used in the legend (before the current value)
        - belonging_fct: The function to use to determine to which category each quote belong
        - investigated_person: The person being investigated
        - filename: The file name under which we should save the figure
    """

    df2 = df.copy()
    df2['cat_boxplot'] = "Other"
    for val in list_values:
        mask = df2.apply(lambda x: belonging_fct(val, x[col]), axis=1)
        df2.loc[mask, 'cat_boxplot'] = str(val)

    fig = go.Figure()
    for i, val in enumerate(list_values):
        color = px.colors.qualitative.Set3[i + 2]
        # Manually makes the fill color more transparent by creating the string rgba(x,x,x,x) required by plotly
        # instead of the string rgb(x,x,x) given by the color variable
        color2 = color[0:3] + 'a' + color[3:len(color) - 1] + ',0.5)'
        fig.add_trace(
            go.Box(y=df2[df2['cat_boxplot'] == str(val)]['compound'],
                   marker=dict(color=px.colors.qualitative.Set3[i + 2]),
                   fillcolor=color2, name=f'{val}'))
    fig.update_layout(
        title={
            'text': f'Mean compound score of quotes talking about {investigated_person} by {col_name}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend_title=col_name,
        xaxis_title=col_name,
        yaxis_title="Compound score"
    )
    fig.show()
    
    # Save plot
    if(filename!=''):
        fig.write_html(filename)

def plot_cluster_pie(df_people_assigned_cluster, cluster_id, filename=''):
  """
    Plot a pie chart of distribution for a cluster.

    Parameters:
      - df_people_assigned_cluster: The dataframe from which we need to compute the pie chart
      - cluster_id: Cluster on which we plot the distribution (-1 if global distribution)
      - filename: The file name under which we should save the figure
  """
  def get_pie(df_people_assigned_cluster, attribut, head=0, prop=0.05, func=lambda x: x[0].upper() + x[1:], **kargs):
    """
    Return a pie chart of distribution for a attribut.

    Parameters:
      - df_people_assigned_cluster: The dataframe from which we need to compute the pie chart 
      - attribut: Attribut
      - head: Number of element not in Other group (0 if use prop)
      - prop: Minimal proportion to be display
      - func: Function to rename the columns
    """
    from group import count_different_values
    df = count_different_values(df_people_assigned_cluster, f'speaker_{attribut}').sort_values('count', ascending=False)
    attribut = attribut[0].upper() + attribut[1:]
    df['index'] = df.index

    # Get the number of significant groups 
    if(head == 0):
      total = df['count'].sum()
      head = df['count'][df['count']/total > prop].count()

    # Groups label with a small proportion in Other
    df_head = df.head(head)
    dt_tail = df.tail(-head)['count'].sum()
    if(dt_tail > 0):
      df_head = df_head.append({'index': f'Other {attribut}', 'count':dt_tail}, ignore_index=True)

    # Rename the labels
    df_head['index'] = df_head['index'].apply(func)
    return go.Pie(labels=df_head['index'].tolist(), values=df_head['count'].tolist(), name=attribut, rotation=-45, **kargs)

  # Get people assigned to cluster `cluster_id`
  df = df_people_assigned_cluster[df_people_assigned_cluster['assigned_cluster'] == cluster_id] if (cluster_id >= 0) else df_people_assigned_cluster
  
  # Plot all distribution
  fig = make_subplots(
      rows=2, 
      cols=2,
      specs=[[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]],
      subplot_titles=('Gender', 'Continent', 'Occupation', 'Nationality')
      )
  gender_map = {
      'male' : 'Men',
      'female' : 'Women'
      }
  fig.add_trace(get_pie(df, 'gender', func=lambda x: (gender_map[x] if x in gender_map else x[0].upper() + x[1:]),
    legendgroup = '1', legendgrouptitle={'text':"Gender"}), 1, 1)
  fig.add_trace(get_pie(df, 'continent', head=5,
    legendgroup = '2', legendgrouptitle={'text':"Continent"}), 1, 2)
  fig.add_trace(get_pie(df, 'occupation', head=5,
    legendgroup = '3', legendgrouptitle={'text':"Occupation"}), 2, 1)
  fig.add_trace(get_pie(df, 'nationality',
    legendgroup = '4', legendgrouptitle={'text':"Nationality"}), 2, 2)
  fig.update_layout(title={
            'text': f'Distribution of cluster {cluster_id}' if (cluster_id >= 0) else 'Global distribution',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
  fig.show()

  # Save plot
  if(filename!=''):
    fig.write_html(filename)