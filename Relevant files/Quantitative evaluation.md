# Quantitative evaluation

* M. Marx
* 2021-03-30

>For some the hell on earth, for others a clear heaven in which progress is always measurable.
 
 
#### abstract

How and why evaluation is done in an MSc IS Data Science thesis.

### Data Science

DS is about discovering patterns in data useful for making accurate predictions on unseen data.

So, if you do DS, you will create something that does something given a piece of data (a prediction usually). Now you, your supervisor at the internship and your examinator want to know how well your creation works. So you show that using an evaluation on **new unseen data containing the correct prediction (label or value)**. In many cases this is **hand labelled gold standard, or golden truth** data.

**If you do not have such data, evaluation is nearly impossible. Almost never is it feasible to create such a dataset by yourself in these 3 months that you have.**

### Format

The way you present your quantitative evaluation follows common practice, and can easily and nicely be done using scikit-learn and seaborn. You want to show how good your model works. So (for a classifier, for regression the story is comparable):

* P,R, F1 scores for each class
* and the averages (indicate whether it is macro or micro)
* PR curve or curves if you compare multiple models
* error analysis based on a confusion matrix

### An extra dataset

Yes, please find one, next to that set provided by your internship. Best is open source, with published results and scores of course. So you know what your model should aim for. Even if your great lovely model performs poorly on your internship data but somewhat in line with the outside data set, you can defend of course. Then an error analysis with some cunning hypotheses on the causes of that strange mismatch are in place of course.

**This extra set gives you piece of mind, and if something goes wrong, a direction to look for the reasons.** So, really it is worth to find one. For yourself, for your grade, and for your examiner.

And really, once you have your stuff in place, running an experiment on that extra set is not that much extra work.

### What if I do unsupervised learning?

Often you do clustering then. Simply giving some clustering quality scores is not enough. We really want to see whether the clusters are useful for the intended application (thus, do they correspond to "natural" groups), and how good the boundaries are. Again, most often you need hand labelled data to assess this.

### I do not do something like that

Of course, an A/B test is also possible  in suitable cases. And there are also other testing/evaluation setups. Your best compass is **to follow a great example** found in an article with many citations/published in a top place.

### My case is different, I do not need golden data

OK, convince yourself, and your examiner (UvA supervisor) **before** you start please.

I see simply one way to do this: by providing a **convincing alternative.** Recall that a data-evaluation as described above is up to a certain point **objective**. Convince yourself that your alternative is that too.

#### No go

* prototype
* proof of concept
* a great methodology to do something without doing it

