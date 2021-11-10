# Project : Milestone 2

## Abstract
These days, machine learning are more and more used on found data, that is data created naturally by the human kind and not during an experiment. Quotebank is an example of such data and can potentially contain a lot of informations about what people think and how they feel over time. 

Our main goal will be to determine what are the sentiments (positive/negative) of different subgroups of the population and how it evolves over time.

We will start by shedding light on the hidden biases in our dataset that can influence the results of our study. 

Then, we will determine whether the sentiments in the quotes of the dataset are postively or negatively biased and perform the same analysis on some  subgroups. Subgroups can be defined according to several criteria, e.g.  age, origin or sex of authors. 

Furthermore, we will see how the sentiments in these subgroups evolves over time.

## Research questions
Our research questions are the following:
- Are the quotes distributed evenly over the years and over the authors ?
- How can we evenly group different quotes by author from the dataset ?
- How are the sentiments distributed among different groups and over the years ?
- Do people tend to talk positively / negatively about other people in the quotes ?

## Subgoals
To be able to answer our research questions, we defined different subgoals that will help us to make conclusions

- Exploratory data analysis :
  - Do many quotes miss an author ?
  - Do many quotes output a possible speaker with a probability less than 0.5 ? 
  - Are the quotes equally distributed over time ? Are there spikes for the number of quotes corresponding to particular events ?
  - What is the distribution of the number of quotes per author ?

- Clustering of the groups :
  - Which clusters can we make depending on the people characteristics ?
  - Are the specified groups equally represented in the dataset ? What are their distributions ?
  - Are our characterization of groups meaningful ? I.e if we change this characterization, does it change our conclusion/interpretation in a significant way ? For instance, suppose we quantised the age of people into ranges such as [0-25], [26-65], [66-100]. Does changing the ranges to [0-30], [31-70], [71-100] significantly change our results or not ?  
  - Is the number of authors in each group balanced ? Is the number of quotes per author balanced ? 
  
- Sentiment analysis :
  - What are the sentiment (positive, negative, neutral) of each quote in the dataset ?
  - Is the reported sentiment really significant in average in the different subgroups we created over the years ?
  - Is there a significant trend where the authors talk negatively/positively in a specific year ? 
  - Are there significant trends in the sentiments of the quotes inside each group (e.g. People over 70 always talking negatively) ?
  - How does the sentiments of a particular author or subgroup evolve over time ? 

- Named Entity Recognition inside the quotes : 
  - Count the number of times each person appears in the quotes. What is the distribution over all the people mentionned ?
  - Who are the people subjects in the quotes and which group do they belong to ? 
  - Are negative/positive quotes talking about someone else ?
  - Are there trends where authors belonging to a certain group write negative/positive quotes about people belonging to another group ? 

  
## Additional datasets
In addition to the given dataset, we will use a dump from Wikidata in order to extract information about the quoter from the Quotebank dataset (for instance the age, sex and origin). We will probably use the dump provided by the teaching staff.
  
## Methods

- Data cleaning: 
We thought of multiple ways to tackle this task : 
    - For our purposes, we want to cluster people according to personal characteristics (sex, age, origin, etc). Therefore, if a quote doesn't have any author, we will have to remove this quote. 
    - The quotes where the author is cited in it is likely to be misclassified. Indeed, it is unlikely that the author will speak of her/him at the third person. As a consequence, we will remove these quotes.
- Clustering of groups:
We will cluster the authors in different ways based on their characteristics. For each subgroup we form, we can compute the mean sentiment of this subgroup for a specific year. Then, we can compare them using T-tests to check if the mean is significantly different between the subgroups. 
- Sentiment analysis: 
We will use [a sentiment analysis classifier from the NLTK library](https://www.nltk.org/api/nltk.sentiment.vader.html). It takes a text (one of our quotes) as input and outputs the probabilities of the quote to be positive, neutral or negative.
- Named Entity Recognition:
In order to extract the names quoted in our dataset, we will use natural language processing with the NLTK library, in combination with an other library called spaCy, a Named Entity Recognition tool. The method we will follow is described [here](https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da). This model should be able to output the name of the people mentionned in the quotes (if any) and extract those containing at least one person to perform analyses.  

## Timeline and Organisation
Cyrille :
- 27.11-04.12 :
  - Extract the sentiment of each quote using [VADER Sentiment Intensity Analyzer from NLTK](https://www.nltk.org/api/nltk.sentiment.vader.html)
  - Compare the main sentiments over the years.
- 04.12-11.12 :
  - Detect main sentiments in different the subgroups of authors.
  - Observe how these groups sentiments evolved over the years and try to interpret them.
- 11.12-17.12 : Data story, writing the final notebook.

Alessio:
- 27.11-04.12 :
- 04.12-11.12 :
- 11.12-17.12 : Data story, writing the final notebook.

Florian :
- 27.11-04.12 : 
    - Extract names of quoted people using NLTK and Spacy.
    - Count the number of quotes where the name of each person appears.
- 04.12-11.12 : 
    - Analyse sentiments of quotes where names appears vs no names appears.
    - Extract quotes where these persons also are quoters.
    - Analyse sentiments of quotes where quoters are quoted and vice versa.
- 11.12-17.12 : Data story, writing the final notebook.

Robin :
- 27.11-04.12 :
    - Exploratory data analysis.
    - Find other unvalide quots.
- 04.12-11.12 :
    - Generate plots and histograms for a representation.
- 11.12-17.12 : Data story, writing the final notebook.
