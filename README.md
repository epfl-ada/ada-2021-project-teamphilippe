# Project : Milestone 2

## Abstract
These days, machine learning are more and more used on found data, that is data created naturally by the human kind and not during an experiment. Quotebank is an example of such data and can potentially contain a lot of informations about what people think and how they feel over time. 

Our main goal will be to determine what are the sentiments (positive/negative) of different subgroups of the population and how it evolves over time.

We will start by shedding light on the hidden biases in our dataset that can influence the results of our study. 

Then, we will determine whether the sentiments in the quotes of the dataset are postively or negatively biased and perform the same analysis on some  subgroups. Subgroups can be defined according to several criteria, e.g.  age, origin or sex of the authors. 

Furthermore, we will see how the sentiments in these subgroups evolves over time.


## Research Questions
We will split our research questions into the four main parts of the project. 

- Exploratory data analysis :
  - Do many quotes miss an author ?
  - Do many quotes output a possible speaker with a probability less than 0.5 ? 
  - Are the quotes equally distributed over time ? Are there spikes for the number of quotes at certain dates or periods corresponding to particular events ?
  - What is the distribution of the number of quotes per author ?

- Clustering of the groups :
  - Which clusters can we make depending on the people characteristics ?
  - Are the specified groups equally represented in the dataset ? What are their distributions ?
  - Are our characterization of groups meaningful ? I.e if we change this characterization, does it change our conclusion/interpretation in a significant way ? For instance, suppose we quantised the age of people into ranges such as [0-25], [26-65], [66-100]. Does changing the ranges to [0-30], [31-70], [71-100] significantly change our results or not ?  
  - Is the number of authors in each group balanced ? Is the number of quotes per author balanced ? 
  
- Sentiment analysis :
  - What are the sentiment (positive, negative, neutral) of each quote in the dataset ?
  - Is the reported sentiment (negative/positive) really significant in average in the different subgroups we created ?
  - Is there a significant trend where the authors talk negatively/positively in a specific year ? 
  - Are there significant trends in the sentiments of the quotes inside each group (e.g. People over 70 always talking negatively) ?

- Named Entity Recognition inside the quotes : 
  - Do many quotes have as subject a person ? I.e. do the quotes give an opinion about someone ?
  - Count the number of times one person appears in all the quotes. What is this distribution over all the people mentionned in the quotes ?
  - Who are the people (if any) subjects in the quotes and which group do they belong to ? 
  - Are negative/positive quotes talking about someone else ?
  - Are there trends where authors belonging to a certain group write negative/positive quotes about people belonging to another group ? 


  
## Additional datasets
In addition to the given dataset, we will use Wikidata in order to extract information about the quoter from the Quotebank dataset (for instance the age, sex and origin). As the whole entity dump from the Wikidata is a file of 70GB, we are not sure (for the moment) whether we will download the whole dump and keep only what we want or if we will first create a list of all the names in the dataset and simply query Wikidata API to get the information we need.
  
## Methods

- Data cleaning: 
For the data cleaning part, we already thought of multiple ways to tackle this task : 
    - For our purposes, we want to cluster people according to personal characteristics (sex, age, origin, etc). Therefore, if a quote doesn't have any author, we will have to remove this quote. 
    - The quotes where the author is cited in it is likely to be misclassified. Indeed, it is unlikely that the author will speak of her/him at the third person. As a consequence, we will also remove these quotes.
- Sentiment analysis: 
For this part of the project, we will use [a sentiment analysis classifier from the NLTK library](https://www.nltk.org/howto/sentiment.html). It takes a text (one of our quotes) as input and outputs the probabilities of the quote to be positive, neutral or negative.
- Named Entity Recognition:
In order to extract the names quoted in our dataset, we will use natural language processing with the NLTK library, in combination with an other library called spaCy, a Named Entity Recognition tool. The method we will follow is described [here](https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da). This model should be able to output the name of the people mentionned in the quotes (if any) and extract those containing at least one person to perform analyses.  

## Timeline and Organisation
  
Cyrille :
- 27.11-04.12 :
  - Use the VADER Sentiment Intensity Analyzer from NLTK (https://www.nltk.org/api/nltk.sentiment.vader.html)
to predict whether a quote is positive/negative/neutral, for each quote in the dataset.
  - Compare the main sentiments over the years.
- 4.12-11.12 :
  - Put things together with the "clustering of people" part to detect main sentiments in different groups of authors.
  - Observe how these groups sentiments evolved over the years and try to interpret them.
- 11.12-17.12 : Data story, writing the final notebook.