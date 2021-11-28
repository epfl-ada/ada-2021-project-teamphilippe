# Project : Milestone 2
# Are the opinions in news quotes related to the success of a company or entrepreneur ?

## Abstract (to be improved into something more convincing)

How does the success of someone's career relate to the opinions of other people ? Can we estimate whether the 
career of a public person is rather at its lowest or its highest only given the quotes mentioning this person 
or her/his companies, etc over time ? Also, can we clearly identify "types" of people (e.g. political orientation, etc) systematically
talk negatively/positively about this person ? Let's take Elon Musk for example. Few years ago, Tesla and SpaceX were not in a very good shape and we could imagine that
quotes about him or his companies were expressing rather negative opinions. But is it really what the data tell us ?
We will look at the quotes mentioning Elon Musk or its companies and try to identify the different cluster of 
people that are criticizing him or supporting him. Throughout the README, we take Elon Musk as example but this easily generalizes to other personalities (such as Mark Zuckerberg).

## Research questions

- How do the sentiments in the quotes mentioning Elon Musk or his companies over time ?
- Among people criticizing/supporting Elon Musk, can we identify specific groups of people ? For example,
  are the people criticizing him mainly old, or coming from specific locations ?
- Are the opinions of other people correlated with the actual success of his career/his companies ?
- Are the opinions of other people about Tesla (or SpaceX) correlated with the opinions about Musk himself ? I.e.
  is Elon Musk viewed by the public "only" through his companies ?
- Optional : Can we identify authors whose opinions have great influence on the stock prices of the companies ?
- Optional : Do the above analysis also apply to personalities from other domain, such as the politics and Trump ?

## Sub-tasks

- Filter the quotes talking about Elon Musk, Tesla, SpaceX using Spacy.
- Determine the sentiments in these quotes using NLTK.
- Find the characteristics of the authors of these quotes, to be able to group them later on.
- Try to identify different groups of people among the authors criticizing vs supporting him.
- Aggregate the stock values of Tesla by month (to be verified).
- Plots the relations of the stock values and the evolution of the opinions about Elon Musk and also
  the number of quotes made about him during each month.
- Try to identify specific authors (or group of authors) whose opinions influence a lot the future stock prices
  of e.g. Tesla.
- Differentiate the quotes talking about Tesla/SpaceX vs Elon Musk itself. Determine whether the opinions
  in each of these groups evolve similarly over time.


## Additional datasets

In addition to the given dataset, we will use:
- Wikidata parquet file provided in order to extract information about quoters from the Quotebank dataset (for instance the age, sex and origin).
- A dataset of stock prices coming from nasdaq.com (for instance [for Tesla](https://www.nasdaq.com/market-activity/stocks/tsla/historical)) for each company owned by Elon Musk.

## Methods
 
- Sentiment analysis:
We will use [a sentiment analysis classifier from the NLTK library](https://www.nltk.org/api/nltk.sentiment.vader.html). It takes a text as input and outputs the probabilities of the quote to be positive, neutral or negative.
- Named Entity Recognition:
In order to extract the names quoted in our dataset, we will use NLP with the NLTK library, in combination with another library called spaCy, a Named Entity Recognition tool.
  The method we will follow is described [here](https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da). This model should be able to output the name of the people mentionned in the quotes (if any) and extract those containing at least one person to perform analyses.
- We plan on finishing the exploratory data analysis and implementing the other parts at the beginning of milestone 3.
  We will apply these towards the middle/end of milestone 3 on the cleaned data.

## Timeline and Organisation
Cyrille :
- 27.11-04.12 :
  - Extract the sentiment of each quote using [VADER Sentiment Intensity Analyzer from NLTK](https://www.nltk.org/api/nltk.sentiment.vader.html).
  - Compare and visualize the main sentiments over the years.
- 04.12-11.12 :
- 11.12-17.12 : Data story, writing the final notebook.

Alessio:
- 27.11-04.12 :
  - Prepare visualizations for the groups of authors in order to identify clusters.
- 04.12-11.12 :
- 11.12-17.12 : Data story, writing the final notebook.

Florian :
- 27.11-04.12 : 
    - Filter the quotes about Elon Musk, Tesla using NLTK and Spacy.
- 04.12-11.12 :
- 11.12-17.12 : Data story, writing the final notebook.

Robin :
- 27.11-04.12 :
    - Add all attributes of the authors (from Wikidata).
    - Exploratory visualizations for these new attributes.
- 04.12-11.12 :
- 11.12-17.12 : Data story, writing the final notebook.
