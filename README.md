# Project : Milestone 3
# Are the opinions in news quotes related to the success of a company or entrepreneur ?

## Abstract 

With the world getting more and more connected, in particular the domain of news and journalism, we get access to opinions of huge amount of people. These opinions can express negative, positive or simply neutral sentiments about a subject or person. Public personalities like Mark Zuckerberg are particularly exposed to criticism, being positive or negative depending on the current progress or discoveries about their companies. But do the opinions in the quotes really reflect the rise of fall of someone' career ? Moreover, do the opinions of specific groups of people indicate an increasing in the company' success ?

## Research questions

- How do the sentiments in the quotes mentioning Mark Zuckerberg or his companies over time ?
- Among people criticizing/supporting Mark Zuckerberg, can we identify specific groups of people ? For example,
  are the people criticizing him mainly old, or coming from specific locations ?
- Are the opinions of other people correlated with the actual success of his career/his companies ?
- Optional : Can we identify authors whose opinions have great influence on the stock prices of the companies ?
- Optional : Do the above analysis also apply to personalities from other domain, such as the politics and Trump ?

## Sub-tasks

- Filter the quotes talking about Mark Zuckerberg or Facebook using Spacy.
- Determine the sentiments in these quotes using NLTK.
- Find the characteristics of the authors of these quotes, to be able to group them later on.
- Try to identify different groups of people among the authors criticizing vs supporting him.
- Aggregate the stock values of Facebook by month (to be verified).
- Plots the relations of the stock values and the evolution of the opinions about Mark Zuckerberg and also
  the number of quotes made about him during each month.
- Try to identify specific authors (or group of authors) whose opinions influence a lot the future stock prices
  of e.g. Facebook.
- Differentiate the quotes talking about Facebook vs Mark Zuckerberg itself. Determine whether the opinions
  in each of these groups evolve similarly over time.


## Additional datasets

In addition to the given dataset, we will use:
- Wikidata parquet file provided in order to extract information about quoters from the Quotebank dataset (for instance the age, sex and origin).
- A dataset of stock prices coming from nasdaq.com (for instance [for Facebook](https://www.nasdaq.com/market-activity/stocks/fb/historical) for each company owned by Mark Zuckerberg.

## Methods
 
- Sentiment analysis:
We will use [a sentiment analysis classifier from the NLTK library](https://www.nltk.org/api/nltk.sentiment.vader.html). It takes a text as input and outputs the probabilities of the quote to be positive, neutral or negative.
- Named Entity Recognition:
In order to extract the names quoted in our dataset, we will use NLP with the NLTK library, in combination with another library called spaCy, a Named Entity Recognition tool.
  The method we will follow is described [here](https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da). This model should be able to output the name of the people mentionned in the quotes (if any) and extract those containing at least one person to perform analyses.
- We plan on finishing the exploratory data analysis and implementing the other parts at the beginning of milestone 3.
  We will apply these towards the middle/end of milestone 3 on the cleaned data.
  
## File structure of the project
```
├── helpers                             
      ├── exploration.py                # Functions to visualise quote features of the dataset 
      ├── extract.py                    # Functions to perform Named Entity Recognition to extract quotes about Mark Zuckerberg and Facebook
      ├── group.py                      # Clusters, distance computations, pre-processing of quotes and people related functions
      ├── group_visualisations.py       # Functions to visualise grouping on attributes and clusters (Plotly and seaborn)
      ├── regression.py                 # Functions to perform regression analysis automatically with statsmodels as well as visualise the response of input variables with the outcomes
      ├── sentiment.py                  # Functions to predict, aggregate and visualise sentiments in the quotes
      ├── sentiment_visualisations.py   # Functions to visualise sentiment (Plotly and seaborn)
      ├── stock.py                      # Functions to handle and visualise stock data
      ├── utility.py                    # Functions to handle the whole Quotebank data set (stream processing functions), preprocess it and sample from it
      ├── wikidata.py                   # Wikidata related functions to add speaker information to the quotes with an identified speaker
├── TeamPhilipe-Project-Notebook.ipynb  # Notebook containing the execution of all the functions, comments and visualisations
├── requirements.txt                    # List of all the packages (and versions) needed to run our project
└── README.md
```
## Package installation
To install all the required packages, you can simply run 
```
pip install -r requirements.txt
```
in the cloned repository.
## Timeline and Organisation
Cyrille :
- 27.11-04.12 :
  - Extract the sentiment of each quote using [VADER Sentiment Intensity Analyzer from NLTK](https://www.nltk.org/api/nltk.sentiment.vader.html).
  - Compare and visualize the main sentiments over the years.
- 04.12-11.12 :
  - Load and aggregate the stock prices data.
  - Look at correlations with the polarity scores.
  - Regression analysis on groups
  - Prepare the website, start writing the introduction, skeleton, etc for the data story.
- 11.12-17.12 : Data story, writing the final notebook.

Alessio:
- 27.11-04.12 :
  - Prepare visualizations for the groups of authors in order to identify clusters.
- 04.12-11.12 :
  - Create custom K-medioid and automatically extract common features within clusters.
  - Regression analysis on different groups.
- 11.12-17.12 : Data story, writing the final notebook.

Florian :
- 27.11-04.12 : 
    - Filter the quotes about Mark Zuckerberg using Spacy.
- 04.12-11.12 :
    - Generalize extraction method to filter quotes from any person/company.
    - Extract what entities were found in each quotes.
- 11.12-17.12 : Data story, writing the final notebook.

Robin :
- 27.11-04.12 :
    - Add all attributes of the authors (from Wikidata).
    - Exploratory visualizations for these new attributes.
- 04.12-11.12 :
    - Create method to extract the continents and group the Academic degree 
    - Create custom K-medioid and plot common features within clusters.
- 11.12-17.12 : Data story, writing the final notebook.
