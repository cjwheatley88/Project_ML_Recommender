#Fortunately, the Harvard X Data Science capstone course has provided the necessary code to initialize both the training and validation data sets.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)

library(caret)

library(data.table)

dl <- tempfile()

download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, validation)

edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Verifying data

#Training data

tibble(rows = nrow(edx), variables = ncol(edx))

data.frame(variable = names(edx),
           class = sapply(edx, typeof),
           first_values = sapply(edx, function(x) paste0(head(x, n = 3),  collapse = ", ")),
           row.names = NULL) %>% 
  tibble()

#example method to assess NAs with for loop.

for (i in 1:ncol(edx)) {
  na <- sum(is.na(edx[,..i]))
  print(na)
}

#Test data

tibble(rows = nrow(validation), variables = ncol(validation))

data.frame(variable = names(validation),
           class = sapply(validation, typeof),
           first_values = sapply(validation, function(x) paste0(head(x, n = 3),  collapse = ", ")),
           row.names = NULL) %>% 
  tibble()

for(i in 1:ncol(validation)) {
  na <- sum(is.na(validation[,..i]))
  print(na)
}

#Objective | Strategy

#Objective:

#Develop and test a movie recommendation model able to predict ratings of movie [i] for user[u]. With accuracy measured through a RMSE < 0.86490. 

#Strategy:  
  
#First examine the relationship between the following variables:
#'ratings' [dependent variable] and; 'userId', 'movieId' and 'genres' [independent variables].
#The reasoning for choosing only these variables; is for brevity and based off an intuition that in a significant volume of reviews.
#Each user movie and genre should detail a generalized bias relative to the mean rating.
#Thus, given new observations with the same independent variables.
#A model can be formed to compute [add the bias terms to a mean] and return a probabilistic estimate of a rating [y_hat].  

#Variable Analysis | Visualization  

#Variable: rating
  
#Frequency distribution of ratings.

hist(edx$rating, xlab = "Rating")

#Ratings histogram tabulated

edx %>% group_by(rating) %>% 
  summarize(count = n()) %>% 
  arrange(., desc(count)) %>% 
  mutate(proportion = round(count/sum(count),2))

modeTrain <- 4

#To summarize what we have found with the 'rating' variable. The mode is 4 and mean is `r mean(edx$rating)`.
#Ratings between whole numbers are less frequent then their whole number equivalent. 
#This variable can be utilized in a supervised learning model to provide real outcomes for training a hypothesis.
#As such, it may be necessary to convert this variable into a factor or category for optimization purposes. 

#Variable: userId
  
#Magnitude of unique users.

N_UserTrain <- length(unique(edx$userId))

N_UserTest <- length(unique(validation$userId))

tibble(N_UserTrain = N_UserTrain, 
       N_UserTest = N_UserTest, 
       Delta = N_UserTrain - N_UserTest)

#The table above depicts the number of unique users in both the training data set [69978] and the test data set [68534].
#Also highlighting the difference [1344] between each.  
  
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color = "black") +
  scale_x_log10()

#The above right skewed plot highlights 'outlier' users, at the higher and lower end of the number of reviews.
#Regularization may be useful to penalize predictions from users with the largest variance from the mean.  

#Variable: movieId  
  
#Magnitude of unique movies.

N_MoviesTrain <- length(unique(edx$movieId))

N_MoviesTest <- length(unique(validation$movieId))

tibble(N_MoviesTrain = N_MoviesTrain, 
       N_MoviesTest = N_MoviesTest, 
       Delta = N_MoviesTrain - N_MoviesTest)


#The table above depicts the number of unique movies in both the training data set [10677] and the test data set [9809].
#Also highlighting the difference [868] between each. 

edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color = "black") +
  scale_x_log10()

#Similar to userId, movieId contains outliers which may also need to be regularized.

#Variable: genres
  
#Magnitude of unique genres.

edx %>% 
  summarize(genres = n_distinct(genres))

#For curiosities sake, let's look at the genres with the most reviews.

edx %>% 
  group_by(genres) %>% 
  summarize(count = n()) %>% 
  arrange(., desc(count)) %>% 
  top_n(., n = 10)
  
#Analyzing the 'genres' variable depicts 797 unique categories within the 'edx' data-set. People love Drama... 


#Hypothesis | Method

#A Naive Bayes method will be utilized to form the model hypothesis. 
#This approach starts with the mean rating for all training reviews and adds bias terms in an iterative fashion.
#Assessing with each addition the accuracy of the model. 
#As stated in the objective, accuracy will be measured through RMSE calculations between a training set and one cross validation set.
#Regularization will be applied where necessary.

#Stratify Data

#I will split the training data into a training data-set and a cross-validation (CV) data-set. [.9 | .1 respectively]

set.seed(1, sample.kind = "Rounding")

index <- createDataPartition(y = edx$rating, times = 1, p = .1, list = FALSE)

train <- edx[-index,]

temp <- edx[index,]

#To avoid our model producing "NA"s; 
#we must ensure the same categorical data is both in the training and cross-validation data-sets.

cv <- temp %>% 
  semi_join(train, by = "movieId") %>% 
  semi_join(train, by = "userId") %>% 
  semi_join(train, by = "genres")

removed <- anti_join(temp, cv)

train <- rbind(train, removed)

rm(removed, temp, index)

# Iteration 1 - Mode and Mean Prediction.

#Initial model utilizing mode only.

modeOnly <- RMSE(cv$rating, modeTrain)

#Initial model utilizing mean only.

meanOnly <- RMSE(cv$rating, mean(train$rating))

results <- tibble(Model = c("Mode","Mean"),
       RMSE = c(modeOnly, meanOnly))

results

#Mean looks to be a more accurate constant for prediction.

# Iteration 2 - Adding Genre.

#Let's add a bias term for Genre as depicted below.

#Mean + Genre

muTrain <- mean(train$rating)

genre_effect <- train %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - muTrain))

pred_1 <- cv %>% 
  left_join(genre_effect, by = "genres") %>%
  mutate(y_hat = muTrain + b_g)

muPlusGenre <- RMSE(pred_1$y_hat,cv$rating)

results <- results %>% add_row(Model = "MeanPlusGenre", 
                               RMSE = muPlusGenre)

results

#With the second iteration - our model has increased accuracy. 

#Iteration 3 - Adding userId.

#Now let's add another bias term - userId. 

#Mean + Genre + userId

user_effect <- train %>% 
  left_join(genre_effect, by = "genres") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - muTrain - b_g))

pred_2 <- cv %>% 
  left_join(genre_effect, by = "genres") %>% 
  left_join(user_effect, by = "userId") %>% 
  mutate(y_hat = muTrain + b_g + b_u)

results <- results %>% add_row(Model = "MeanPlusGenre_PlusUser", 
                               RMSE = RMSE(pred_2$y_hat, cv$rating))

results

#Iteration 4 - Adding movieId.

#Now let's add another bias term - movieId

#Mean + Genre + userId + movieId

movie_effect <- train %>% 
  left_join(genre_effect, by = "genres") %>% 
  left_join(user_effect, by = "userId") %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - muTrain - b_g - b_u))

pred_3 <- cv %>% 
  left_join(genre_effect, by = "genres") %>% 
  left_join(user_effect, by = "userId") %>% 
  left_join(movie_effect, by = "movieId") %>% 
  mutate(y_hat = muTrain + b_g + b_u + b_i)

results <- results %>% add_row(Model = "MeanPlusGenre_PlusUser_PlusMovie", 
                               RMSE = RMSE(pred_3$y_hat, cv$rating))

results

#The model is becoming more accurate, however we are still a ways off from the performance objective. 
#Let's add regularization into the mix.

#Adding regularization
#Certain parameter values have significantly less frequent observations, and since we are trying to generalize an average prediction for each dimension.
#These outlier values can add unwanted variability into our predictions.
#As such we will add the penalty term Lambda to our hypothesis, reducing variability for each bias term.
#To optimize our hypothesis and select the most accurate value for lambda.
#We will iterate values from 4 to 10; in increments of .25. 
#Plotting our results and selecting the lambda value which returns the minimum RMSE.  

lambdaList <- seq(4, 10, .25)

rmseFinal <- sapply(lambdaList, function(l){
  
  mu <- mean(train) 
  
  genre_effect <- train %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating - muTrain)/(n() + l))
  
  user_effect <- train %>% 
  left_join(genre_effect, by = "genres") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - muTrain - b_g)/(n() + l))
  
  movie_effect <- train %>% 
  left_join(genre_effect, by = "genres") %>% 
  left_join(user_effect, by = "userId") %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - muTrain - b_g - b_u)/(n() + l))
  
  predFinal <- cv %>% 
  left_join(genre_effect, by = "genres") %>% 
  left_join(user_effect, by = "userId") %>% 
  left_join(movie_effect, by = "movieId") %>% 
  mutate(y_hat = muTrain + b_g + b_u + b_i) %>% 
  pull(y_hat)
  
  return(RMSE(predFinal, cv$rating))
  
})

#Optimization Plot

plot_tibble <- tibble(rmse = rmseFinal, lambda = lambdaList)

plot_tibble %>% ggplot() +
  geom_point(aes(x = lambda, y = rmse))

results <- results %>% add_row(Model = "MeanPlusGenre_PlusUser_PlusMovie_Regularized", 
                               RMSE = min(rmseFinal))
results

#Hypothesis Testing

lambda <- lambdaList[which.min(rmseFinal)]

rmseTest <- function(l){
  
  muFinal <- mean(edx$rating)
  
  b_g <- edx %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - muFinal)/(n() + l))
  
  b_u <- edx %>% 
    left_join(b_g, by = "genres") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - muFinal - b_g)/(n()+ l))
  
  b_i <- edx %>% 
    left_join(b_g, by = "genres") %>% 
    left_join(b_u, by = "userId") %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - muFinal - b_g - b_u)/(n() + l))
  
  pred <- validation %>% 
    left_join(b_g, by = "genres") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_i, by = "movieId") %>% 
    mutate(pred = muFinal + b_g + b_u + b_i) %>% 
    pull(pred)
  
  return(RMSE(pred, validation$rating))
  
}

rmse <- rmseTest(lambda)

rmse

results <- results %>% add_row(Model = "Validation", RMSE = rmse)

results
