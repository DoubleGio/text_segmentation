# EDA Exploratory Data Analysis

## Why?

In your thesis you will work with data. In most cases you will *predict* the value on one variable based on the values of (lots of) other variables. 

In your EDA you are going the *understand* your data. Because with this understanding you can better analyse, debug, explain, improve, ... your system.

Everything you do in your EDA needs to come from this desire to *understand*. You focus on those aspects of your data which are central to your research questions, to your methods, and to your desired outcomes.

### Data cleaning

EDA and data cleaning go hand in hand. While doing the EDA you discover anomalies, missing values, outliers, mistakes (typos/..) that can be repaired, etc, etc. 

#### Provenance

Of course you clean data and repair mistakes, but make sure it is totally transparent. Both *what* you do, *why* you do it, and *what effect* it has on your corpus.

## What?

You want to know your data at several levels:

* corpus
* variables by themselves
* interaction between variables

### Corpus

* How many instances, how many variables (`df.shape`)
* Nr Missing values for each variable.
* Type of each variable
* If applicable, clusters
* ...

#### pre variables

* Sometimes you want to describe your corpus also *before* you actually have variables that you can give to scikit learn.
* Time
* When data is text or video, lengths, all kind of counts (voc size eg, hapax count, etc, etc)

### Univariate analysis

* How does the population look when you focus on just one aspect (variable)
* `df.describe()`
* `sns.boxplot`
* `sns.displot`, histograms, `kde`
* possibly factored by values of the to be predicted variable
* priors


#### Baseline

Based on this analysis you should be able to deteremine baseline scores for your predictor. E.g., if we know that the male/female ratio is .8 in our corpus, I can make a classifier with 80% accuracy without any thinking or machine learning or anything. I need to beat that figure of course. 

You want to know and understand that number long before you start.

### Multivariate analysis

* How do variables interact? Often that means, how do they *correlate*?
    * `pd.crosstabs`, groupby, pivot tables for categorical variables
    * correlation for numerical ones
* `sns.pairplot`
* Should you remove variables or not? 

## How?

EDA can be done brilliantly in `pandas` combined with `seaborn`. Read [section 4.14 of the Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.14-Visualization-With-Seaborn.ipynb) for lots of ideas. 

Create one or a few EDA/cleaning/normalization notebooks with good markdown explanantions, simple nicely refactored code, and great looking visuals.

### Normalization

Is also something that may be needed, but always should be based on a good understanding (=EDA) of your data.

## And my thesis?

Many students see EDA as a waste of time. It is not. Besides the advantages sketched above I could think of these:

* often this is the most useful and used part of your thesis from the perspective of the company
* easy and nice to write part of your thesis with cool (why not even interactive) visuals
    * always nice to do when you have that writers block
* impressive visuals for your presentation (and thesis cover)
* now you?


## (Peer Review) Criteria for a good EDA notebook

1. Start with a numbered  overview of your research questions.
2. Relate all that you do to your RQs or subRQs, name them explicitly.
3. **Explain** in words what specific parts of your EDA  means for your research.
4. At the end, the reader should
   1. understand your data in general
   2. understand how you can answer your RQs based on this dataset
   3. be able to replicate your data cleaning and preprocessing, and understand why these steps were taken
   4. have a good feeling from looking at stunning (maybe interactive?) visuals which give a clear picture of the data
   5. be enthousiastic to read more about this research
5. Best is if a reader can **run** your EDA notebook, but that means the data must be available and accessible. If that is not possible, make sure that the version on github contains all output.