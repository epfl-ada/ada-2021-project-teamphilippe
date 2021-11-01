# Project : Milestone 2

## Abstract
These days, machine learning and data are more and more used to make decisions and studies. To avoid making 
discriminatory decisions against certain group of people or draw negatively influenced results, it is therefore crucial that people using these data are 
aware of the potential biases in the data.


The goal is to shed light on the hidden biases of the dataset that can greatly influence the results of a study
that would use this dataset to train, e.g. ML models or compute statistics. In particular, we would like to 
analyse whether sentiments in quotes of certain groups of people are rather positive or negative and how this evolves
over time. We will look at different subsets of people by characterizing them according to :
- Sex of the author
- Origin of the author
- Age of the author


## Research Questions
- Exploratory data analysis :
  - Are the specified groups equally represented in the dataset ? What are their distributions ?
  - Do many quotes miss an author ?
  - Are the quotes equally distributed over time ? Are there spikes corresponding to particular events ?
- Sentiment analysis :
  - What are the sentiment (positive, negative, neutral) of each of the quotes in the dataset ?
- Named Entity Recognition inside the quotes :
  - Do many quotes have as subject a person ? I.e. the quotes gives an opinion about someone.
  - Who are the people (if any) subjects in the quotes and to what group do they belong ? (Can use a library called spaCy which is a Named Entity Recognition
  tool, see here https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da)
- Clustering of the groups :
  - How can we cluster the people according to their characteristics ?
  - Are there significant trends in the sentiments of the quotes inside each group ?
  - Are there trends where authors belonging to a certain group write negative/positive quotes about persons belonging
  to another group ?