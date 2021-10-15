## ML Conceptual

* references:
  * https://towardsdatascience.com/data-science-interview-guide-4ee9f5dc778

* explain the difference between supervised and unsupervised learning
  * supervised learning separates data into pre-determined human defined categories based on labels
  * unsupervised learning separates data without the use of labels.
* what is bias?
  * difference between average prediction in our model and the correct prediction
* what is variance?
  * variability of model output for a given observation
* what is the bias variance tradeoff?
  * model is too simple, can't accurately understand dataset
  * model too complex, can't generalize to new data points
* what is overfitting?
  * overfitting means a model has learned its parameters too specifically to your particular data set which means it is not going to be able to effectively generalize to unseen data
  * **high variance low bias**
    * its *Too* good at our learning problem
  * how to address?
    * regularization
    * more training data
    * parameter tuning
* what is underfitting?
  * underfitting means a model has not effectively learned parameters to model your particular data set so it is not accurately solving your problem of interest
  * **low variance high bias**
    * its not able to accurately separate our data
  * how to address?
    * more training data
    * more complex model
    * ensemble learning
* what is the curse of dimensionality?
  * bigger dims, data becomes sparse
  * and # of model configurations grows exponentially
  * more features doesn't necessarily improve performance
    * cant actually hurt performance
  * simple calc like 'distance' func becomes exponentially more difficult
* when to use classification vs regression?
  * classification separates data into camps of categorical labels
  * regression classifies based on a numerical attribute
* what is F1?
  * harmonic mean of precision and recall
* what is precision?
  * number of true positives / true positives + false positives
* what is recall?
  * number of true 
* why do we sample?
* what is bootstrapping?
  * iteratively resampling your dataset in order to estimate population metrics
  * when to use it?
    * very useful when you have a constrained/limited sized dataset but still want to carry out advanced analysis
* what is a loss function?
  * a loss function measures the ability of your model to accurately make predictions by comparing your results to their true labels
    * example: cross entropy loss for classification tasks
* what is ensemble learning?
  * what is bagging?
* how to summarize your data?
  * how do you discover outliers in your data?
  * what assumptions can you make e.g. when can we say it's "normal"
* what is the central limit theorem?
* what is the law of large numbers?
* models to know...
  * k means
  * svm
  * decision tree
  * hierarchical clustering
  * logistic
  * linear
    * what assumptions do you make when using linear models?
    * what features?
  * knn
  * topic modeling
  * naive bayes
* what is regularization?
  * L1 vs L2?
  * when to use?
* diff bt linear and logistic?
* explain bayes theorem
  * what is prior?
  * what is posterior?
* diff bt parametric and non parametric model?
  * parametric example
  * non parametric example
* how do we evaluate a model?
* explain a statistical significance test
  * t - test rundown (a/b)
* what is type I
* what is type II
* explain cross validation
  * leave one out
  * k-fold
* what is stemming?
* what is lemmatization?
* when would you want to keep stop words?
* explain word embeddings.
* what is chi squared
* what is ANOVA
* what is a confidence interval?
* what is a p value?
* what is statistical power?
* why is BERT so good?
    * transformers
    * attention
* how do you reduce dimensionality of textual data?
    * PCA
      * explain PCA
    * LDA
* how do you deal with an imbalanced data set, like credit card fraud detection?
* what is homoskedasticity
* what is a (non linear) activation function
  * a non linear activation function transforms inputs to a non linear space which allows a neural network to patterns from a non linear version of the input
  * this allows it to make sophisticated decisions across boundary lines that aren't constrained by linearity
* what is A/B testing