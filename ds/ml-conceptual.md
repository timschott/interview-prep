## ML Conceptual

* references:
  * https://towardsdatascience.com/data-science-interview-guide-4ee9f5dc778

## Data Exploration

* how to summarize your data?
  * 5 number summary
  * leverage `ntile()` / `avg()` / `min()` / `max()` in SQL, `summary()` in R, `.describe()` on a pandas df
* how do you discover outliers in your data?
  * sort by desc, asc
* what assumptions can you make e.g. when can we say it's "normal"
* how do you detect nulls in your data?
  * R can use `is.na`, we can fill in gaps
  * `ozone_rep = airquality['Ozone'].fillna(0.0)`
  * `data.replace('?',0, inplace=True)`
  * SQL will automatically remove NULL values from aggregations, but you can some them by checking `IS NULL`
* should you replace nulls in your data? what are your options?
  * you can replace them with 0, but this could throw off average calculations
  * you could consider them to be truly null, and remove those rows from imputations down the line
    * but this could throw off your modeling performance because you're discarding an entire
  * you could replace with the median of the column
    * depends on the type of data that you are working with - in some cases, this doesn't make sense
  * you can perhaps split one column with some nulls into 2 columns that is instead a binary indicator
  * creating / filling in known as "data imputation"
  * you could replace with random samples, bounded by the lower/upper bounds of the dataset
* how do you deal with an imbalanced data set, like credit card fraud detection?
  * generate synthetic data
  * oversample the minority class when forming training / testing blocs
  * use a very sensitive cost function to artificially balance the training process
    * you can do this w/ SVM
  * use a metric besides accuracy (F1 that takes false negatives into account for example)

## Model Evaluation

### Concepts

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

### Regularization

* what is regularization?
  * L1 vs L2?
    * L1 = LASSO
      * least absolute squares
      * adds an absolute value of penalty
      * because of this, it can completely remove a coefficient by setting its value to zero
    * L2 = Ridge Regression
      * adds a squared magnitude of penalty
  * when to use?
    * regularization prevents a single feature from being too large during model training. it can be used when you need to prevent overfitting and want your model to generalize better.
    * we would use L1 LASSO during feature selection because it can completely drop unneeded coefficients

### Metrics

#### F1, Precision, Recall

* what is F1?
  * harmonic mean of precision and recall
* what is precision?
  * number of true positives / true positives + false positives
* what is recall?
  * number of true positives / true positive + false negative
* what is a residual?
  * difference between a predicted value and a true value
* What is RMSE / RMSD?
  * root mean square error / root mean square deviation
  * standard deviation of the sum of prediction errors
  * `√(Σ(ŷi – yi)^2)) / n)`
  * on the same scale as output 

#### Error and Error Measures

* where does error come from?
  * [todo]
* What is SE?
  * standard error = variation of sample statistics
  * formula: `SE = s / sqrt(n)`
    * s - population std. dev
    * n - sample size
    * used when we have a group of random samples from a normally distributed dataset 
* What is SSE?
  * sum of squares error
  * the sum of the squared differences between the predicted data points (`ŷi`) and observed data points (`yi`)
  * `SSE = Σ(ŷi – yi)^2`
  * residuals
* What is Mean Square Error?
  * SSE divided by population size
  * `MSE = (1/n) Σ(ŷi – yi)^2`
  * "RMSE" is just the square root of this (standardizes)
* What is SSR?
  * sum of squares regression
  * the sum of the squared differences between the predicted data points (`ŷi`) and the mean of the response var (`ȳ`)
  * `SSR = Σ(ŷi – ȳ)^2`
  * 
## Machine Learning

* explain the difference between supervised and unsupervised learning
  * supervised learning separates data into pre-determined human defined categories based on labels
  * unsupervised learning separates data without the use of labels.

### Models [todo]

* models to know...
  * k means
  * svm
  * decision tree
  * random forest
  * hierarchical clustering
  * logistic
  * linear
    * what assumptions do you make when using linear models?
    * what features?
    * what is homoskedasticity
  * what is ANOVA [tod[]]
    * analysis of variance
    * what are the measures it produces
  * knn
  * topic modeling
  * naive bayes
  * neural network
    * simple
    * fully-connected
    * CNN
    * LSTM

### Strategies and Discussion

* very high level: describe the steps of a learning pipeline
  * [todo]
* what is ensemble learning?
  * combining different models in order to produce more robust outcomes
  * bagging
    * bootstrap and build multiple classifiers, one for each sample, then combine classifiers
  * boosting
    * "sequential learning"
    * XGBoost
    * ADAboost
* diff bt parametric and non parametric model? [todo]
  * parametric example 
  * non parametric example
* explain cross validation [todo]
  * leave one out
  * k-fold
* explain grid search [todo]
* what is the curse of dimensionality?
  * bigger dims, data becomes sparse
  * and # of model configurations grows exponentially
  * more features doesn't necessarily improve performance
    * cant actually hurt performance
  * simple calc like 'distance' func becomes exponentially more difficult
* when to use classification vs regression?
  * classification separates data into camps of categorical labels
  * regression classifies based on a numerical attribute
* what is a (non linear) activation function
  * a non linear activation function transforms inputs to a non linear space which allows a neural network to patterns from a non linear version of the input
  * this allows it to make sophisticated decisions across boundary lines that aren't constrained by linearity
* diff bt linear and logistic?
  * linear regression is used to separate data into classes on a continuous scale
    * uses a linear function, generalized form of w = xt + b, produces a real number value
  * logistic regression is used to separate data into categorical groups
    * uses a logistic function, the sigmoid, to estimate the *probability* of belonging to a group (bt 0 and 1)
* explain bayes theorem
  * bayes theorem lets us leverage information we know about an event to determine the probability that a conditional event we have not observed will take place
  * what is prior?
    * prior probability is some evidence we use to model an event before it takes place
  * what is posterior?
    * the probability distribution conditional on evidence from the survey, the outcome that you are after
* what is a loss function?
  * a loss function measures the ability of your model to accurately make predictions by comparing your results to their true labels
    * cross entropy loss for NNs
    * hinge loss for SVM 
  * typically described in terms of the performance on a *single* data point
* what is a cost function?
  * similar to loss function, but typically refers to the average loss of the whole training 
  * Mean Square Error
* what is an objective function?
  * most general term for the function you optimize in training
  * MLE 
  * Negative Log Likelihood 
* what is back propagation?
* what is pooling?
* what is convolution?
* what is dropout?
* what is "fully connected"?

## Probability

* diff bt independent and mutually exclusive?
  * two events that are independent if prior knowledge of A occurring has no bearing on B's outcome
    * P(A and B) = P(A) * P(B)
    * P(A|B) = P(A)
    * P(B|A) = P(B)
    * for example, drawing marbles but replacing them after each pull
  * two events are mutually exclusive if the occurrence of A prevents B from taking place
    * P(A or B) = P(A) + P(B)
    * P(A and B) = 0
    * for example, team A wins and team B wins

## Statistics

### Sampling

* what is the central limit theorem?
  * if you draw large, random samples from a populations, the means of those samples will be distributed normally around the population mean
  * you can game this out by noting the variance of the sample mean drops to 0 as the number of samples becomes very large
  * for example, 1000 fair coin flips can be modeled by a normal dist N~(500, 250)
* what is the law of large numbers? [todo]
  * sample gets better as sample size increases
* why do we sample?
* what is bootstrapping?
  * iteratively resampling your dataset in order to estimate population metrics
  * when to use it?
    * very useful when you have a constrained/limited sized dataset but still want to carry out advanced analysis


### Hypothesis Testing

#### Procedure

* explain a hypothesis test
  * test rundown
  * when to use a binomial test
    * when we have very few samples of a Bernoulli trial
      * / proportions
  * when to use a z-test
    * used when we know the population variance, and sample size is >= 30 (CLT)
  * when to use a t-test
    * used when we do not know the population variance, and sample size is >= 30 (CLT), and our samples are normally distributed
* compare t-distribution to normal-distribution
  * t-dist is more spread out
  * used we don't know standard dev
  * produces wider confidence intervals than z-dist 
    * (for small-ish sample sizes)
* which t-test do i use?
  * one-sample should be used to compare the *mean* of 1 sample to a *known* value
  * two-sample should be used to compare *two* means from *two* samples
    * if equal sample sizes and equal variances (assumed): common population error pooling
    * if not: welch's t test - https://en.wikipedia.org/wiki/Welch%27s_t-test
* how to calculate a z score?
  * when **X** is a single random variable and we are pulling from a dist:
  * `z = (x-μ)/σ`
  * when we are *sampling*, we are randomly drawing variables from some distribution `~N(μ, σ)`
  * this new distribution will have the same mean, but a different variance
  * so, a z score from a *sample distribution* `~~N(μ, σ^2/n)` is:
  * `z = (𝑋-μ)/(σ/√𝑛)`
  * remember that the point of the z score is to give us a *standardized* score
    * that's why we have that denominator change -- we are taking the square root of all of `σ^2/n`, as thats our variance, and square root of variance is standard dev
* how to calculate a t score?
  * general form: sample differences / standard error
  * one sample t test:
    * `t = (𝑋- μ)/(s/√𝑛)`
    * where 𝑋 is the sample mean
    * where s is the *sample* standard dev
  * two sample t test:
    * `t = (𝑋1 - X2) / [standard error]`
* how to determine significance?
  * is the retrieved test statistic larger (/smaller) than the critical value
  * if so, reject null hypothesis
* how to calculate interval of acceptance from sig level
  * [todo]

#### Hypothesis Testing Conceptual

* what is a critical value
  * the threshold that - if exceeded - leads to the conclusion that the difference between the sample mean and the (hypothesized) population mean is large enough to reject the null hypothesis
  * = to value whose probability of occurrence is <= alpha
    * note that you might have to adjust if you have a 2 tailed test
  * popular critical value: alpha is .05, z = 1.96
* what is type I
  * type I error is a false positive
  * the error of rejecting the null hypothesis when it is true
  * ie, we accept an alternative hypothesis even though it can be attributed to chance!
  * its probability is Alpha
  * (telling a man he is pregnant)
* what is type II
  * type II error is a false negative
  * when we fail to reject the null hypothesis when it is false
  * ie, we fail to observe a statistical difference when there is one!
  * (telling a quite-pregnant woman she is not pregnant)
  * probability is Beta
* what is statistical power
  * the probability of *not* making a type II error, of failing to accept the alternative hypothesis when there is enough evidence to accept it is not occurring by random chance
    * 1 - Beta
* what is a significance level
  * the significance level represents the probability of rejecting the null hypothesis when it is true
* what is a confidence interval?
  * gives us a measure of how confident we are that the statistic of interest falls in that interval 
  * x% of confidence intervals we generate will capture the parameter
  * we are x% confident in the process used to generate our interval
  * how to calculate?
  * (sample mean) +/- margin of error
* what is margin of error?
  * critical val * standard deviation/standard error
    * where critical value is z-score at sig level!
    * standard dev or standard error depending on what you are working with
* what is a p value?
  * the probability of obtaining test results at least as extreme as what was observed
  * "probability this difference occurred by random chance"
  * the amount of area not covered by the accumulation of the dist wrt the retrieved test statistic
* what is A/B testing
  * A/B testing is just another way of setting up a hypothesis test.
  * average revenue per user, Gaussian
  * `t.test(data1, data2, var.equal=TRUE)	`
  * use a z-test, t-test, depending on the information you have about your samples
  * most of the time we're not going to know the population standard deviation because this is real-world data
  * so the t test comes in handy for those situations
* what are degrees of freedom
  * number of independent pieces of info
  * typically, N-1

## NLP

* what is stemming?
  * [todo]
* what is lemmatization?
  * [todo]
* what is name entity recognition?
  * [todo]
* what is coreference resolution?
  * [todo]
* in today's world, what is the most robust way to do sentiment analysis?
* when would you want to keep stop words?
  * in certain applications like coreference resolution or part of speech tagging, stop words could be very important
  * for example we would want to keep determiners like 'an' 
* explain word embeddings.
  * word embeddings provide a nuanced representation of words
  * they serve as the initial input to many NLP problems
  * to construct word embeddings, packages like GLoVE and word2vec 
  * [todo]
* what is chi squared
  * tests whether there is a statistically significant difference between observed and expected frequencies
  * for NLP: could use chi squared on tf-idf scores
* why is BERT so good?
  * word piece
  * pre-training
  * bidirectional
  * masking
  * next sentence prediction
  * attention
  * transformers
  * fine-tuning
* how do you reduce dimensionality of textual data?
    * PCA
      * explain PCA
      * [todo]
    * LDA
      * matrix decomposition
      * removing unneeded columns
      * [todo]
* compare an contrast BERT and GPT-3
  * [todo]
* explain how a transformer works
  * [todo]
* you just created an NLP model. what are some ways you can see how well it performs?
  * mixture of standard error / recall measures (accuracy, F1, ROC) and task specific like the stanford question and answer data set (SQuAD)
  * you should also try to look at how efficient it is 
    * - how many parameters have you added versus baseline? 
    * is there a significant gain for x amt of increase in training time / corpus size?
      * if it takes 2 weeks to train your model, it's going to be hard to iteratively retrain it with new data
  * also, can it generalize? can it be applied to other tasks>
* what is zero-shot learning?
  * leveraging a language model by directly running test cases w/ no "fine-tuning" / domain-specific training
* what is prompt-based learning?
  * using the same style of question and answer learning that language models often use in their training phase when you are going through your task of interest
  * adjusting your objective like sentiment analysis of part of speech tagging to be solvable via "prompts"
    * masking, permutation, english + non-english pairings, NER
* sig testing in NLP, why do we have to be careful?
  * don't use a t-test if we aren't working with an average measure
  * don't assume data is iid
    * (text data from same author is not independent)
  * mixing k fold and significance tests
    * you need to be calculating a p-val / checking null hypoth for each fold
  * using a t-test for classification
    * there are better tests for this
  
## Resume

* Work
  * ETL for Janes?
    * transformed CMS data (Rich text, particularly) from roughly HTML/JSON to XML
    * added metadata / other info from external APIs
    * DITA compliant
    * then dropped in s3
* Thesis
  * tokenization?
    * word, sentence, paragraph
    * by hand, rules based / regex
    * probably should have used a package, there
  * feature engineering
    * 31 features like, punctuation per line, syllables per word, etc
  * Decision Trees
    * used random forests for feature selection via the variable importance metric
    * a random forest was used because ensemble learning is an effective way to combine the output of many decision trees
      * prevents overfitting
      ```R
          m <- randomForest(new_df[,-31], new_df$label2, 
                        sampsize = round(0.8*(length(new_df$label2))),ntree = 500, 
                        mtry = sqrt(30), importance = TRUE)
      ```
  * SVM
    * used SVM because I had just learned that in class and wanted a binary separation of my data
      * used LOOCV since I had a pretty small data set
      ```R
      folds <- cvFolds(NROW(sub), K=50)
      results <- c(0)
      names <- c(0)
      for(i in seq(1, 50)){
        # run LOOCV each time 50 times and see what the average accuracy is. 
        
        train_temp <- sub[folds$subsets[folds$which != i], ] #Set the training set

        test_temp <- sub[folds$subsets[folds$which == i], ] 

        temp_fit <- svm(label2 ~., data = train_temp, kernel = "linear",
                      cost = 1)
        test_grid_temp <- predict(temp_fit, newdata = test_temp)

        mat <- confusionMatrix(test_grid_temp, test_temp$label2)
        if(mat$overall[1] != 1){
          print(i)
          names <- append(names, big_boy[i,1])
        }
        results<-append(results, mat$overall[1])
      }
      ```

  * graphing?
    * created plots using ggplot2 and matplotlib
    ```python
    for i,x in enumerate(container):
        row = np.array(x)
        row = row.astype(float)
        y = np.arange(1, len(row)+1, 1)
        plt.figure(figsize=(9,9))

        plt.plot(y, row)
        plt.title(titles[i])

        plt.savefig(titles[i] + '_MATTR.png', dpi=500)
        plt.gcf().clear()
        plt.show()
    ```

* Research
  * logistic
    * predicting case outcomes?
    * realized that this wasn't quite the level of analysis we wanted to work on
  * name entity recognition
    * try to distinguish which parties were being references in cases
  * topic modeling
    * mallet
  * metadata work
    * figuring out questions like, given a date, return who was on the court / who was chief justice
* GPT-2 (SBOTUS)
  * interested in using court corpus to make a "bot" for each justice
  * did not have enough data to do this reliably, though, so just made one "composite" justice model
  * added extra data by using the Oyez API 
    * for each case, pull in the oral arguments for each participating justice
    * this left me with a huge amount of data for each justice - all their sentences
  * [todo]
* Twitter BOT [todo]
* NLP project
  * [how is it going] [todo]
* Classes?
  * applied NLP
  * next semester, hopefully TA

## Current, Interesting Papers

* wilmot and keller, 2021
  * Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories
  * http://arxiv.org/abs/2109.03754
  * salience detection with RAG
* bender et al, 2021
  * Stochastic Parrots
  * https://dl.acm.org/doi/pdf/10.1145/3442188.3445922
  * are lang models too big?