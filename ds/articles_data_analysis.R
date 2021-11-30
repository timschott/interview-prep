## Article Data analysis

library(dplyr)
library(tidyr)
library(ggplot2)

article_data <- read.csv("data/articles_data.csv")

# task: predict whether an article will be a "top" article or not.
# measure of error: accuracy, precision, recall, F1

# drop unneeded first index
article_data <- article_data[, seq(2,ncol(article_data))]

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



