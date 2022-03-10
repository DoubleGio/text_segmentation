# Baseline ML algorithm plus evaluation

This week you create a **strong but simple baseline** which 

* gives you insight in achievable performance
* gives insight in the types of errors made
* gives ideas on better/different representations of your data and/or selection of features
* provides you a "pipeline" for doing complete ML experiments, including an evaluation that provides insights, "evidence" on trustworthiness of your learned model and (promising) directions for  possible improvements

### Great example

[This blogpost on **simple** baselines](https://blog.insightdatascience.com/always-start-with-a-stupid-model-no-exceptions-3a22314b9aaa) is a great start. I hope it helps you realize that doing something absolutely non-fancy as a start is in fact a great start!

If you need a worked out example, continue with [this post by the same author on text classification](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e), accompanied by a well organized [Jupyter notebook](https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb).

In fact, such a well worked out notebook is really worth the time and energy to make. You can be proud, easily share your results with peers, and receive tips and feedback. 

The nice thing is that it shows that **what you are doing makes sense, and leads to (OK, sometimes they are small) improvements.**

**Small addition to that notebook:** also try an even simpler representation: letter n-grams, for n in 2,3,4. And, as a way to add a bit of the structure of the text, you may add *word bi or even trigrams* to your tokens (unigrams). 

## Evaluation

* Use the ideas stated in this notebook ([PCA](https://github.com/jakevdp/PythonDataScienceHandbook/blob/8a34a4f653bdbdc01415a94dc20d4e9b97438965/notebooks/05.09-Principal-Component-Analysis.ipynb) plotting to see the separatedness of the classes), and also these in the [seaborn chapter of the Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.14-Visualization-With-Seaborn.ipynb) to get a feeling for the problem **before** you start training a model.
* Make an extensive "rich evaluation output" setup that you can reuse again and again:
    * P,R,F1 values for all your classes
    * confusion matrix
    * error plots, PR curves
    * ...
* Use the ideas in this notebook for inspecting **important features** for all of your classes. Does it make sense? Are you not overfitting?

 
## Criteria for Peer review

1. Is the task clear? 
    2. Also, did you get some idea on how hard it is?
2. Is the model simple/stupid enough, and the results acceptable (i.e. better than majority class)?
3. Are the "obvious" improvements tried out, and evaluated?
4. Does the presented model make sense? 
    * are "important" features for the model also acceptable for you? 
5. Do the errors make sense? Is it indicated where improvements could or should be made?
6. **Was it a pleasant and fun experience to peer review this notebook?** Did you learn something from it, even if you do a quite different research topic?

## Eh, what if I do *unsupervised* learning?

* I like the treatment of these techniques in the Data Science Handbook a lot: eg  [PCA](https://github.com/jakevdp/PythonDataScienceHandbook/blob/8a34a4f653bdbdc01415a94dc20d4e9b97438965/notebooks/05.09-Principal-Component-Analysis.ipynb) and [Gaussian Mixture models](https://github.com/jakevdp/PythonDataScienceHandbook/blob/8a34a4f653bdbdc01415a94dc20d4e9b97438965/notebooks/05.12-Gaussian-Mixtures.ipynb).
* Still you want to evaluate, but if you really have no labels, that seems impossible. So you need at least something.
* If you have some data "clustered" in classes, you can see how well your cluster algorithm caught these classes.
* There are also scores which say something about the intrinsic quality of the clustering, like the silhouette score and the calinski_harabasz score.
* Some useful links:
    * <https://towardsdatascience.com/how-to-evaluate-unsupervised-learning-models-3aa85bd98aa2> is really introductory
    * <https://arxiv.org/pdf/1905.05667.pdf> is quite exhaustive and goes quite deep.
    * Speciaal voor NLP toepassingen: <https://dl.acm.org/doi/pdf/10.5555/2140458.2140463>
    * [And as the last one, something special](https://stats.stackexchange.com/questions/195456/how-to-select-a-clustering-method-how-to-validate-a-cluster-solution-to-warran) Not for the faint-hearted.
