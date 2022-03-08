# Research Questions 

* also called RQs.

## Why do you need them?

1. To structure your thesis; your work; your research; your everything you do in these 3 months of thesis project.
2. To make sure your reader (examiner) **always knows** why she is reading something. 
    3. Your RQs are the backbone, or maybe the whole skeleton, of your thesis.
    4. They provide for a smooth reading experience, a clearly build structure, and a convicing argument.

## Their form

* I am fond of a tree like structure of (sub) research questions. 
* my standard setup is 
    * depth 3
    * branching factor 3
    * Numbered RQs
* So RQ, SRQ1, SRQ2, SRQ3, SRQ1.1, SRQ1.2, ....
* The **root** is a very general  question making clear what you will do.
* Your sub RQs provide the **evidence** with which you will answer the root RQ in your conclusion.
* Your **leaf sub RQs** are super concrete, and a reader can expect the *shape of the answer* (eg, a PR curve, a set of F1 scores for difefrent systems plus an indication if differences with a baseline are significant, etc)
* All of your RQs should be easy to read and understand. The higher in the tree, the more they are understandable by non-experts.

## Their use

1. For every paragraph, every subsection, every section in your thesis it should be clear to which (S)RQ it is connected.
2. Everything you do in these three months need to be connected to a (S)RQ.
3. And vice-verse, you need to answer all your (S) RQs.


## How to make (S)RQs?

1. Yeah, I know that is really difficult.
2. But, hey, it is not a one directional process. 
3. In the beginning, the main RQ must be clear of course. 
4. And that gives rise to certain types of evidence that you need to gather by doing research/experiments. 
5. So you "break that up" into these sub RQs, and subsubRQs.
6. But, while doing the research and the experiments it may turn out things are a little different than you thought, so you answer a slightly different question. 
    7. No problem: simply adjust your SRQ so that it is a question **that is answered by what you did**.
8. Also, if you get great new ideas, no problem to add one or more (S)RQs.
    9. Even if they diverge from your main RQ.
    10. As long as it is interesting and closely related, you can always do this.


## What are not RQs

1. All questions for which others have provided the answer already.
    2. That is, things you can find in the literature.
3. To find the answers to *your RQs*, **you** must do, well eh, .... yes *research*!
    4. experiments, work with data, code, think, play, experiment, you know science
4. Questions that can trivially be answered, like "Is it possible...." (yes, everyting is possible, well almost everything).

### Examples of tricky cases
    
1. *What is the optimal....*
    2. That is a pretty hard question to answer if it is not constrained very very heavily.
3. *Which AI/ML/... method for my XYZ problem leads to higher profits for my company?*
    4. Can you really test that? With what kind of test design? Will your company allow you to do that?
    5. Maybe that was the *motivation* to start the project for your company, but a motivation is usually not an RQ.


## Check/Homework/Peer review points

1. Try to answer each of your research questions in a really stupid way. If you can, you must reword it.
2. For each RQ, provide the _format_ of the answer, and the _form of the evidence_ which belongs to that answer.
	* E.g. RQ: What is the influence of "aggresive feature selection" for my text classification performance? 
	* **Evidence** Typically a graphic in which you vary the feature selection and show the resulting performance.
		* E.g.: number of features on the x-axis, and F1 score on the y-axis
		* Or, a set of precision-recall curves for a number of feature selection settings
		* In a  good case, an optimal value(range) becomes clear from your grahic (so you also have worse performance on both sides).
3. For all of the things that you have done this week, indicate the RQ for which you have done it. If there is none, stop doing that , or simply create the appropiate RQ ;-)

### Peer reviewers: look extra good at this point

*Peer reviewer: pretend you are an advocate of the devil and find it hard to believe that the student can really answer her questions with her data.*  *Simply try hard to imagine how she will do it with her data. Dare to ask. Dare to question. Dare to probe!* You really help your peer with this!

4. Combine your EDA with your RQs and ask yourself:
    5. Do I really have the data, and the features, and the gold standard train and test sets to answer each of my RQs and SRQs?
    6. Is it **crystal clear** what data I need for each of my SSRQs? And what the quality and the "richness/content" of the data must be?

