# Methodology-experimental setup-results

The template of our MSc Data Science thesis has these three sections. But what material goes where? And should you always have these three?

## Function of each section

### experimental setup

1. Function 1: The user can **replicate** your experiments.
2. Function 2: The user gets a (very) **good idea of your used dataset**.

So in the *description of the data* subsection you paste your most insightful graphs from your EDA, next to the basic statistics on your dataset and descriptions (statistical and/or a population density diagram) of your variables.

In the experimental setup section you give all the *settings* used in your experiments. All needed so that someone with your dataset and your software can replicate your work and obtain more or less (often there are random effects) the same results. Think of

1. data preprocessings steps used
2. hyperparameter settings (for all of them)
3. How you created a train-validate-test split or otherwise did your training and testing.
4.  Exactly how the used metrics are calculated (think of the difference between micro and macro F1 for classification, and how often this is not explicitly stated).

**This section can be quite long. If you want or need to tell more, consider using the Appendix or a reference to a nicely structured notebook in which all experiments are done.**

### Methodology

You have mentioned how others have used the methods you will use in your related work section. Here you will go into more details about them. 

In particular you pay attention to that new little thing, that change, that great idea, that you **add** to the existing method.

The function is that the reader knows what you are using, and how in particular you are using it. 

This is not a textbook section (readers can find that elsewhere, probably better), nor a place to copy/paste difficult intimidating formulas.  

**This section can be quite brief. Use most space for your addition. If there is none, the section can be rather brief.**

### Results

The function is that you give, for each of your research question, the **outcomes** of the experiment corresponding to the question in the form of a table or  a graphic. 

#### Rule of thumb

Structure your section such that the reader should only read these two things, and can savely skip all else:

1. the question
2. the table/figure and the (elaborate) caption.

No need to explain things in words which are already in your table. 

Much need for a perfect caption, perfect labels, smart design of table or figure. So the reader can use 100% of her brain to **understand** the outcomes, not to try to figure out what was meant.

**So this section can be quite brief (in words). You really answer your questions in the conclusion/discussion.**

## Do I really need 3 sections?

No. Sometimes, especially if you have lots of, quite different, experiments/research questions it makes sense to interleave the experimental setup and the result sections, so the reader does not get lost, and need not remember very much.

Good structuring in subsections and maybe subsubsections is then helpful.
