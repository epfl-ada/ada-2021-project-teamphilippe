import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def assigned_categories(df, col_group, group_by_values, belonging_fct=lambda val, lst: val in lst):
    df['assigned_cat'] = None
    for index, val in enumerate(group_by_values):
        mask = df.apply(lambda row: (row[col_group] is not None) and belonging_fct(val, row[col_group]), axis=1)
        df.loc[mask, 'assigned_cat'] = index
    return df


def prepare_df_quotes_regression(df, col_group, groupby_values, period='%Y-%V',
                                 belonging_fct=lambda val, lst: val in lst):
    """
    Prepare a dataframe from the quotes dataframe that contains the
    avg positive and negative scores per period and per group as well
    as the number of quotes per period and per group.

    Parameters:
      df : df to process
      col_group: col name of the df to group by
      groupby_values: list of values to group by in the col_group
      period: string format to group by period
              (by month : '%Y-%m', by week : '%Y-%m-%V')
      belonging_fct: function of belonging to a group value.
        It takes as input a group value and a list of values.

    Returns:
    df_res: df with avg (on period) polarity scores for all groups with the
      number of quotes by period and by group
    assign_dic: dict mapping column names to values of the groups column
    """

    df = assigned_categories(df, col_group, groupby_values, belonging_fct)
    df_res = pd.DataFrame()
    assign_dic = {}

    for index, val in enumerate(groupby_values):
        # Select quotes of that value
        cur_df = df.apply(lambda row: row['assigned_cat'] == index, axis=1)
        cur_df = df[cur_df]

        # Group quotes by period and aggregate them
        cur_df = cur_df.groupby(cur_df.date.dt.strftime(period)).agg(['mean', 'count'])[['neg', 'pos']]

        # Build the columns of the resulting dataframe
        name = col_group + "_" + str(val).replace(', ', '_').replace('(', '').replace(')', '')
        assign_dic[name + "_count"] = index
        assign_dic[name + "_neg_mean"] = index
        assign_dic[name + "_pos_mean"] = index

        df_res[name + "_count"] = cur_df['neg']['count']
        df_res[name + "_neg_mean"] = cur_df['neg']['mean']
        df_res[name + "_pos_mean"] = cur_df['pos']['mean']

    return df_res, assign_dic


def regression_analysis(df_polarity_scores, df_stock, col_regression=None):
  """
  Perform a regression on the predictors:
  - mean polarity scores by groups and period
  with response variable the avg between open and close stock price.

  Parameters:
  - df_polarity_scores: df containing the predictors variables
  - df_stock: df containing the response variable
  - col_regression: predictors to use in the regression. If None, use
      all the variables

  Returns:
  - fitted model
  """
  # Standardize the columns
  df_polarity_scores_std = (df_polarity_scores - df_polarity_scores.mean(axis=0)) / df_polarity_scores.std(axis=0)

  # Append the stock values delayed by 1 week
  df = pd.concat([df_polarity_scores_std, df_stock['avg_op_cl'].shift(-1)], axis=1)

  formula = 'avg_op_cl ~'
  for col in df_polarity_scores.columns:
      if col_regression is None or col in col_regression:
          formula += col + ' + '

  formula = formula[:-3]

  # Linear regression
  model = smf.ols(formula=formula, data=df)
  res = model.fit()
  print(res.summary())

  return res


def standardize_concat(df_polarity_scores, df_stock):
  """
  Standardize the features and concatenate the features
  with the delayed response variable by 1 week (so that
  features of week n are on the same line as the response
  variable of week n+1).

  Parameters:
    - df_polarity_scores: df containing the predictors variables
    - df_stock: df containing the response variable

  Returns:
    - df containg the standarized features with the shifted
      response variable
  """
  # Standardize the columns
  df_polarity_scores_std = (df_polarity_scores - df_polarity_scores.mean(axis=0)) / df_polarity_scores.std(axis=0)

  # Append the stock values delayed by 1 week
  return pd.concat([df_polarity_scores_std, df_stock['avg_op_cl'].shift(-1)], axis=1)


def plot_response_wrt_features(df, x_vars, y_vars):
  """
  Plot each standardized feature against the
  response variable.

  Parameters:
    - df: df containing the features and the response variable
    - x_vars: list of features to plot
    - y_vars: list of of the response variables
  """
  def go(x_vars_tmp):
    sns.pairplot(data=df, x_vars=x_vars_tmp, y_vars=y_vars,
                kind='reg', height=5, aspect=.8)
    plt.suptitle('Avg stock price wrt features', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.plot()

  for i in range(0, len(x_vars), 3):
    go(x_vars[i:i+3])


def scatter_plot_stock_mean(df_regr, df_stock, assign_dic, groupby_value, col_split, shift=-1, col_stock='avg_op_cl',
                            col_stock_title='Average stock price'):
  dfs = []
  for index, val in enumerate(groupby_value):
    all_cols_for_cur_val = [col for col in df_regr.columns if assign_dic[col] == index]
    cur_df = df_regr[all_cols_for_cur_val].copy()

    # Standardize the columns
    cur_df = (cur_df - cur_df.mean(axis=0)) / cur_df.std(axis=0)

    rename = {all_cols_for_cur_val[0]: "Count", all_cols_for_cur_val[1]: "Negative mean",
              all_cols_for_cur_val[2]: "Positive mean"}
    cur_df.rename(columns=rename, inplace=True)
    cur_df.loc[:, 'category'] = index
    dfs.append(cur_df.reset_index())

  all_df = pd.concat(dfs, axis=0).reset_index(drop=True)

  # Append the stock values delayed by 1 week
  df = all_df.join(df_stock['avg_op_cl'].shift(shift), on='date')
  fig, axs = plt.subplots(1, 3, figsize=(20, 5))

  # Plot the 3 wanted graphs
  sns.scatterplot(data=df, x="Count", y="avg_op_cl", hue="category", ax=axs[0], palette='colorblind')
  h, l = axs[0].get_legend_handles_labels()
  axs[0].legend(h, groupby_value, title="Sentiment")
  axs[0].set_ylabel(col_stock_title)
  axs[0].set_title("Relation between count and " + col_stock_title)

  sns.scatterplot(data=df, x="Negative mean", y="avg_op_cl", hue="category", ax=axs[1], palette='colorblind')
  h, l = axs[1].get_legend_handles_labels()
  axs[1].legend(h, groupby_value, title="Sentiment")
  axs[1].set_ylabel(col_stock_title)
  axs[1].set_title("Relation between negative mean and " + col_stock_title)

  sns.scatterplot(data=df, x="Positive mean", y="avg_op_cl", hue="category", ax=axs[2], palette='colorblind')
  h, l = axs[2].get_legend_handles_labels()
  axs[2].legend(h, groupby_value, title="Sentiment")
  axs[2].set_ylabel(col_stock_title)
  axs[2].set_title("Relation between positive mean and " + col_stock_title)
  plt.suptitle(f"Relation between features and stockprices by splitting on {col_split}")
  plt.show()


def plot_regression_results(res, filename=""):
    """
      Gather the computed coefficients of the features and their
      p-values. Plot them using a color code.

      Parameters:
      - res: Regression results object from statsmodels
    """
    # Collect the p-values as well as the coefficients
    df_res = pd.DataFrame()
    df_res['pvalue'] = res.pvalues
    df_res['Coefficient'] = res.params
    df_res['Feature'] = res.params.index
    order = df_res['Feature'].tolist()

    # Plot positive coeff in blue and negative in red
    # Plot statistically non significant at level 0.05 coeffs in gray
    df_res['color'] = ['positive coefficient'] * df_res.shape[0]
    df_res.loc[df_res['Coefficient'] < 0, 'color'] = 'negative coefficient'
    df_res.loc[df_res['pvalue'] >= 0.05, 'color'] = 'insignificant coefficient'

    # Do not keep the intercept
    df_res = df_res[df_res['Feature'] != 'Intercept']
    # Plots the coefficients
    fig = px.bar(df_res, x='Coefficient', y='Feature', orientation='h', hover_data={'pvalue':':.4e', 'Coefficient':':.4e'},
                 color='color', color_discrete_map={'negative coefficient': px.colors.qualitative.Set3[3], 'positive coefficient': px.colors.qualitative.Set3[4], 'insignificant coefficient': px.colors.qualitative.Set3[8]},
                 title='Regression coefficients', category_orders={'Feature': order}, height=700)
    fig.update_layout(title={
      'text': 'Regression coefficients',
      'y': 0.9,
      'x': 0.5,
      'xanchor': 'center',
      'yanchor': 'top'},
        legend_title="Coefficient colors")
    if filename!="":
      fig.write_html(filename)
    fig.show()
