## Article Data analysis

library(dplyr)
library(tidyr)
library(ggplot2)

article_data <- read.csv("data/articles_data.csv")

# task: predict whether an article will be a "top" article or not.
# measure of error: accuracy, precision, recall, F1

# drop unneeded first index
article_data <- article_data %>% select(-c(X))
# how many "top" vs not articles?
# 9161 not
# 1276 yes
# 2 NA
article_data %>% group_by(top_article) %>% summarize (count = n())

# so this leaves us with somewhat of an unbalanced data set. 
# but, 1000+ obs is still a lot to go off of. 

# EDA -- 

# first, explore "bad" rows
colSums(is.na(article_data))

# the rows with top_article missing should just get dropped entirely.
# one is a news story, the other is odd (some weird values)

article_data %>% filter(is.na(top_article))
article_data <- article_data %>% drop_na(top_article)

# moving on, 
# across the dataset, ~1% of the "engagement" metrics are NA.
# can safely replace those with 0

article_data <- article_data %>% replace_na(list(engagement_reaction_count = 0,
                       engagement_comment_count = 0,
                       engagement_share_count = 0,
                       engagement_comment_plugin_count = 0))

# next, let's take a look at each column.
# source_id - where it is published
# note that this is essentially the same as the column that follows it
# source_name
article_data %>% group_by(source_id) %>% summarize (count = n())
# where are the popular stories being published?
article_data %>% group_by(source_id, top_article) %>% summarize (count = n()) %>% arrange(count)
# every ESPN story is a top article ... 
# other than that, the data is fairly balanced
# this means that we shouldn't use source_id as a factor

# author
article_data %>% group_by(author) %>% summarize (count = n()) %>% nrow()

# three types of authors:
# not provided -- oped, some web sources just omit this data
# the sources themselves, ie, when BBC News is listed as an author
# or (for ~ 20% of articles), normal bylines
# either way, can't read too much into this col
# could perhaps engineer "hasRealAuthor" as a binary
article_data %>% 
  group_by(author) %>% 
  summarize (count = n()) %>% 
  arrange(desc(count), author)

# let's make a graph for this -- top 10 most frequent "authors"
article_data %>% 
  mutate(author = ifelse(author == "", "not provided", author)) %>%
  group_by(author) %>% 
  summarize (count = n()) %>% 
  arrange(desc(count)) %>% 
  filter(row_number() <= 10) %>% 
  ggplot(aes(x = reorder(author, -count), y = count)) + 
  geom_bar(stat = 'identity') +
  labs(x='Author',y='Count') +
  theme(axis.text.x = element_text(angle = 45, hjust=1))

# title ....
# to start, are there any duplicated titles?
# with syndication, content "sharing" etc, this may be the case 
article_data %>% select(title) %>% unique() %>% nrow() # 9809
article_data %>% select(title, url) %>% unique() %>% nrow() # 10434
# so, 9809 titles but 10435 articles ... definitely have duplicates
# however -- that is not the best judgement of a truly "duplicated" piece of content
# for the purpose of this exercise, a better judgement is something like URLs.
# because each unique url = a unique touchpoint for engagement (commenting, liking, etc)
# so we will tolerate duplicated titles

# description
# similar story to title.

# as far as how to use title and description, you could engineer features like
# number of words
# number of nouns
# number of entities
# dictionary lookup against "sensational" words (crazy, wild, explosion, etc)
# dictionary lookup if you had access to "trending" topics at that time
# ie an article about the Superbowl published in February will probably get traffic
# however, those may not be practically sig. 

# url 
# a couple duplicated URLs.
# at this point, I'm OK with removing these, for reasons mentioned above
article_data <- article_data %>% distinct(url, .keep_all = TRUE)

# url to image
# here, I'm just interested if this exists or not
# because without it, I'm guessing there is no "lede"
# I would bet lede's drive engagement, so this is something to convert to
# a boolean 
article_data %>% 
  mutate(url_to_image = ifelse(url_to_image == "", "not provided", "provided")) %>% 
  group_by(url_to_image) %>%
  summarize (count = n())

# almost every article has an image URL. 
# to make more useful, could test if these URLs are "active" (ie a connection to them doesnt 404)
# as sometimes CDNs change, people change links, etc.

# published at - what is the date range? 
# 14 calendar days, mostly in september 2019 and a few in october 2019
# this is helpful, because we can assume a fairly consistent user pattern in that time
article_data %>%
  group_by(as.Date(published_at)) %>%
  summarize(count = n())
# i don't think time of day is a helpful indicator here
# in particular because we don't have time zone location for the times
# in addition, stories that get engagement don't necessarily get read over morning coffee
# habits are sporadic, so it's not worth splintering this

# before we start to do analysis, let's randomly partition a test set that is 10% of the original data set.
library(caret)
trainIndex <- createDataPartition(article_data$top_article, p = .9, 
                                  list = FALSE, 
                                  times = 1)
# shuffle the dataframe
article_data <- slice(article_data, sample(1:n()))

train_data <- article_data[trainIndex, ,drop=FALSE]
test_data <- article_data[-trainIndex, ,drop=FALSE]

# now, let's add features to the train data set

# investigate a mentions_trump (in title) column
train_data %>% 
  mutate(mentions_trump = ifelse(grepl('Trump', title, ignore.case = TRUE), 1, 0)) %>%
  group_by(mentions_trump) %>%
  summarize(count = n(), pct = n() * 100 / nrow(train_data))
# ~ 6% of headlines mention trump
train_data <- train_data %>% 
  mutate(mentions_trump = ifelse(grepl('Trump', title, ignore.case = TRUE), 1, 0)) 

# let's start looking at the engagement statistics
# engagement_plugin_comment_count is almost always 0. that one is not helpful
train_data <- train_data %>%
  select(-c(engagement_comment_plugin_count))
# how many articles have "any" "engagement"?
train_data %>%
  filter(engagement_reaction_count != 0 | engagement_comment_count != 0 | engagement_share_count != 0) %>%
  summarize(count = n(), pct = n() * 100 / nrow(train_data))

# 3/4 have "any" engagement. good discrim
train_data <- train_data %>%
  mutate(engaged = ifelse(engagement_reaction_count != 0 | engagement_comment_count != 0 | engagement_share_count != 0, 1, 0))

# some articles don't have "content" because they are video
# let's make a video column that detects when video is used in the URL
train_data %>%
  filter(grepl('video', url, ignore.case = TRUE)) %>%
  summarize(count = n(), pct = n() * 100 / nrow(train_data))

# 7.4 % of examples
train_data <- train_data %>%
  mutate(video = ifelse(grepl('video', url, ignore.case = TRUE), 1, 0))

# Length of the article
# we can figure this out from the [...x chars]
# in the content column
# improved regex 
library(stringr)
train_data %>%
  select(content) %>%
  substr(..)

# [+1744 chars]
test_df <- as.data.frame(train_data[8493,])
c <- test_df$content
m <- regexec("[0-9]+ chars", c)
regmatches(c, m)

no <- "hello tim"
z <- regexec("[0-9]+ chars", no)
length(regmatches(no, z))

test_df %>% 
  select(content) %>% 
  mutate(new_col = stringr::str_extract(content, "([0-9]+) chars")) %>%
  mutate(new_col = ifelse(is.na(new_col), 0, str_sub(new_col,1,nchar(new_col)-6)))

# add content_length col
# all the abbrev are the same length, so not a big deal we don't include that part.
train_data <- train_data %>%
  mutate(content_length = stringr::str_extract(content, "([0-9]+) chars")) %>%
  mutate(content_length = ifelse(is.na(content_length), 0, str_sub(content_length,1,nchar(content_length)-6))) %>%
  mutate(content_length = as.numeric(content_length))

# look at a hist of the non zero entries and non huge entries
train_data %>% filter(content_length > 0 & content_length < 10000) %>% ggplot(aes(x = content_length)) + geom_histogram(bins = 300) 

# now let's use mentions-trump, any_eng, video and char_count in logistic regression.
# make sure those are all factors
train_data <- train_data %>% 
  mutate(mentions_trump = as.factor(mentions_trump)) %>%
  mutate(engaged = as.factor(engaged)) %>%
  mutate(video = as.factor(video)) %>% 
  mutate(top_article = as.factor(top_article))

## Logistic Regression on training data. 
log_reg_with_four_columns <- glm(top_article ~ mentions_trump + engaged + video + content_length, data = train_data, family = "binomial")
summary(log_reg_with_four_columns)

# confidence intervals for the coefficients
confint(log_reg_with_four_columns)

library(aod)
# chi squared for a particular column
wald.test(b = coef(log_reg_with_four_columns), Sigma = vcov(log_reg_with_four_columns), Terms = 2)

# odds ratios per coefficient
# an article that mentions trump is 2.202 times as likely to be a top article!
exp(coef(log_reg_with_four_columns))

# deviance of the model
with(log_reg_with_four_columns, null.deviance - deviance)

# overall p val
with(log_reg_with_four_columns, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

library(pscl)
# mcfadden's R2 is quite close to 0, so this isn't very good.
pR2(log_reg_with_four_columns)

# very small.

# let's do some predictions.
# first, run the test data through the same dplyr routine....
test_data <- test_data %>%
  mutate(mentions_trump = ifelse(grepl('Trump', title, ignore.case = TRUE), 1, 0)) %>%
  mutate(mentions_trump = as.factor(mentions_trump)) %>%
  select(-c(engagement_comment_plugin_count)) %>%
  mutate(engaged = ifelse(engagement_reaction_count != 0 | engagement_comment_count != 0 | engagement_share_count != 0, 1, 0)) %>%
  mutate(engaged = as.factor(engaged)) %>%
  mutate(video = ifelse(grepl('video', url, ignore.case = TRUE), 1, 0)) %>%
  mutate(video = as.factor(video)) %>%
  mutate(content_length = stringr::str_extract(content, "([0-9]+) chars")) %>%
  mutate(content_length = ifelse(is.na(content_length), 0, str_sub(content_length,1,nchar(content_length)-6))) %>%
  mutate(content_length = as.numeric(content_length)) %>%
  mutate(top_article = as.factor(top_article))

test_data$top_article_p <- predict(log_reg_with_four_columns, test_data, type = "response")

library(gmodels)
# very imbalanced data set ...
cross_table <- CrossTable(x = test_data$top_article, y = ifelse(test_data$top_article_p > .5, 1, 0), prop.chisq = FALSE, dnn = c("Actual", "Prediction"))

TP <- cross_table$t[1]
FN <- cross_table$t[2]
FP <- cross_table$t[3]
TN <- cross_table$t[4]
## Recall (Sensitivity)
(TP) / (TP + FN)
## Specificity
(TN) / (TN + FP)
## Precision
(TP) / (TP + FP)
## False Positive Rate
1 - ((TN) / (TN + FP))
## Accuracy
(TP + TN) / (TP + FP + FN + TN)
## F1 Score
2*TP / (2*TP + FP + FN)

summary(train_data$top_article)

# succesfully engineered and created a Logit model.
# however, still running into what i saw last time
# what to do about imbalanced data?
# use the ROSE package.
library(ROSE)
# in practice, you would only want to do this to your training set
# because you don't want to make fake test cases
# note that in order to use this, you can only feed it
# continuous or categorical (ie factor) vars
# so, whatever you were feeding to the model, for instance, use that
# so now try out oversampling + all the numerical features
train_data_rose <- ROSE(top_article ~ mentions_trump + engagement_share_count + content_length, data =  train_data, seed = 42)$data

# this is now balanced
# lets build another model and see if it works better
## Logistic Regression on training data. 
rose_logit <- glm(top_article ~ mentions_trump + engagement_share_count + content_length, data = train_data_rose, family = "binomial")
summary(rose_logit)
pR2(rose_logit)

# adding type = response directly gives us the probabilities instead of summing the coefficients
rose_preds <- predict(rose_logit, test_data, type="response")

cross_table <- CrossTable(x = test_data$top_article, y = ifelse(rose_preds > .5, 1, 0), prop.chisq = FALSE, dnn = c("Actual", "Prediction"))

TP <- cross_table$t[1]
FN <- cross_table$t[2]
FP <- cross_table$t[3]
TN <- cross_table$t[4]
## Recall (Sensitivity)
(TP) / (TP + FN)
## Specificity
(TN) / (TN + FP)
## Precision
(TP) / (TP + FP)
## False Positive Rate
1 - ((TN) / (TN + FP))
## Accuracy
(TP + TN) / (TP + FP + FN + TN)
## F1 Score
2*TP / (2*TP + FP + FN)

varImp(rose_logit)

## ROC / AUC 
library(pROC)
RC <- roc(test_data$top_article,ifelse(rose_preds > .5, 1, 0))
plot(RC, legacy.axes = TRUE)

## Feature Selection (for regression)

# make the biggest model (all "relevant" cols)
modelfull <- glm(top_article ~ mentions_trump + engaged + engagement_share_count + engagement_comment_count + engagement_reaction_count + video + content_length, data = train_data, family = binomial)
# trace gets rid of output
library(MASS)
stepwise_model <- stepAIC(modelfull, direction = "both", trace = FALSE)
varImp(stepwise_model)

# curious, lets see if this does any better!

stepwise_preds <- predict(stepwise_model, test_data, type="response")
cross_table <- CrossTable(x = test_data$top_article, y = ifelse(stepwise_preds > .5, 1, 0), prop.chisq = FALSE, dnn = c("Actual", "Prediction"))
TP <- cross_table$t[1]
FN <- cross_table$t[2]
FP <- cross_table$t[3]
TN <- cross_table$t[4]
## Recall (Sensitivity)
(TP) / (TP + FN)
## Specificity
(TN) / (TN + FP)
## Precision
(TP) / (TP + FP)
## False Positive Rate
1 - ((TN) / (TN + FP))
## Accuracy
(TP + TN) / (TP + FP + FN + TN)
## F1 Score
2*TP / (2*TP + FP + FN)
varImp(stepwise_model)
library(dplyr)
knn_train_data <- train_data %>% select(c(mentions_trump, engagement_reaction_count, engagement_comment_count, engagement_share_count, engaged, video, content_length, top_article))
knn_test_data <- test_data %>% select(c(mentions_trump, engagement_reaction_count, engagement_comment_count, engagement_share_count, engaged, video, content_length, top_article))

## lets just do KNN for practice.
library(class)
knnModel <- knn(train=knn_train_data, knn_test_data, cl=knn_train_data$top_article, k=2)
summary(knnModel)

cross_table <- CrossTable(x=knn_test_data$top_article,y=knnModel,prop.chisq = FALSE, dnn = c("Actual", "Prediction"))
TP <- cross_table$t[1]
FN <- cross_table$t[2]
FP <- cross_table$t[3]
TN <- cross_table$t[4]
## Recall (Sensitivity)
(TP) / (TP + FN)
## Specificity
(TN) / (TN + FP)
## Precision
(TP) / (TP + FP)
## False Positive Rate
1 - ((TN) / (TN + FP))
## Accuracy
(TP + TN) / (TP + FP + FN + TN)
## F1 Score
2*TP / (2*TP + FP + FN)

knn_with_prob <- knn(train=knn_train_data, knn_test_data, cl=knn_train_data$top_article, k=2, prob = TRUE)
probs <- attributes(knn_with_prob)[[3]]
## new metric, log-loss
library(MLmetrics)
LogLoss(y_pred = probs, y_true = as.numeric(levels(test_data$top_article))[test_data$top_article])


