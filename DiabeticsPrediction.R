if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("reshape2", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(matrixStats)
library(reshape2)
library(dplyr)
library(randomForest)
library(gam)



url <- "https://github.com/Bitakhparsa/Diabetics_prediction/raw/main/diabetes2.csv"
download.file(url, destfile = "./data.csv", method="auto")
diabetes1 <- read.csv("data.csv")



#Exploring Data

glimpse(diabetes)
head(diabetes)

any(is.na(diabetes))



diabetes_new <- diabetes %>%
  mutate(Outcome = factor(Outcome),
         BMI = ifelse((BMI==0),
                      median(BMI, na.rm = TRUE), BMI),
         Insulin = ifelse((Insulin==0), 
                          median(Insulin, na.rm = TRUE), Insulin),
         SkinThickness = ifelse((SkinThickness==0),
                                median(SkinThickness, na.rm = TRUE), SkinThickness),
         BloodPressure = ifelse((BloodPressure==0),
                                median(BloodPressure, na.rm = TRUE), BloodPressure),
         Glucose = ifelse((Glucose==0), 
                          median(Glucose, na.rm = TRUE), Glucose)
         )
         
        

# Data Visualization 

diabetes_new %>% group_by(Outcome) %>% summarise(n=n())
diabetes_new %>% ggplot(aes(Outcome)) + geom_bar()


diabetes_new %>%
  filter(Outcome == "1") %>%
  ggplot(aes(BMI)) +
  geom_histogram(bins = 30)


# Distribution of each predictor stratified by outcome.

diabetes_new_melt <- melt(diabetes_new , id="Outcome")

diabetes_new_melt %>% 
  ggplot(aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=Outcome))+
  facet_wrap( ~ variable, scales="free")


#____________________________
y <- diabetes_new$Outcome
x <- diabetes_new %>% select(-Outcome)
class(y)
class(x)
x<- as.matrix(x)



# Scaling the matrix x

x_centered <- sweep(x,2,colMeans(x))
x_scaled <- sweep(x_centered,2,colSds(x), FUN="/")


#Heatmap 

d <- dist(t(x_scaled))
heatmap(as.matrix(d))


#Hierarchical clustering
h <- hclust(d)

plot(h, cex = 0.65, main = "", xlab = "")

groups <- cutree(h,k=3)

split(names(groups), groups)


#Create Diabetes_data, validation, test set and train set 

set.seed(1, sample.kind = "Rounding") 
index <- createDataPartition(diabetes_new$Outcome, times = 1, p = 0.2, list = FALSE)

Diabetes_data <- diabetes_new %>% slice(-index)
validation <- diabetes_new %>% slice(index)

nrow(Diabetes_data)
nrow(validation)


set.seed(1, sample.kind = "Rounding")
ind <- createDataPartition(Diabetes_data$Outcome, times = 1, p = 0.2, list = FALSE)

train_set <- Diabetes_data %>% slice(-ind)
test_set <- Diabetes_data %>% slice(ind)

nrow(train_set)
nrow(test_set)



#Logistic regression model

set.seed(1, sample.kind = "Rounding") 

train_glm <- train(Outcome ~ ., data = train_set, method = "glm", family="binomial")
glm_preds <- predict(train_glm, newdata = test_set)
cf_glm <- confusionMatrix(glm_preds,test_set$Outcome)
       
#Creating a result table
results <- tibble(method ="Logistic regression model", 
                  accuracy =cf_glm$overall["Accuracy"], 
                  Sensitivity =cf_glm$byClass[1] ,
                  Specificity = cf_glm$byClass[2])      
 



#K-nearest neighbors model

set.seed(1, sample.kind = "Rounding") 

train_knn <- train(Outcome ~ ., method = "knn",
                   data=train_set,
                   tuneGrid = data.frame(k = seq(5, 90, 2)) )
train_knn$bestTune

knn_preds <- predict(train_knn, test_set)

cf_knn <- confusionMatrix(data=knn_preds, test_set$Outcome)




#adding result in the table
results <- bind_rows(results,data_frame(method="K-nearest neighbors model",
                                        accuracy =cf_knn$overall["Accuracy"], 
                                        Sensitivity =cf_knn$byClass[1] , 
                                        Specificity = cf_knn$byClass[2] ))
results %>% knitr::kable()




#Random forest model
set.seed(1, sample.kind = "Rounding")
train_rf <- train(Outcome ~ ., data=train_set, 
                  method="rf", 
                  importance = TRUE,
                  nodesize = 1,
                  tuneGrid = data.frame(mtry=seq(3, 9, 1)))

rf_preds <- predict(train_rf, test_set)

cf_rf <- confusionMatrix(data=rf_preds, test_set$Outcome)




train_rf$bestTune

plot(train_rf)

varImp(train_rf)

#adding result in the table
results <- bind_rows(results,data_frame(method="Random forest model",
                                        accuracy =cf_rf$overall["Accuracy"], 
                                        Sensitivity =cf_rf$byClass[1] , 
                                        Specificity = cf_rf$byClass[2] ) )
results %>% knitr::kable()


#Final model

glm_preds_final <- predict(train_glm, newdata= validation)

cf_glm_final <- confusionMatrix(data=glm_preds_final,validation$Outcome)


#adding result in the table
results <- bind_rows(results,data_frame(method="Final model",
                                        accuracy =cf_glm_final$overall["Accuracy"], 
                                       Sensitivity =cf_glm_final$byClass[1] , 
                                       Specificity = cf_glm_final$byClass[2] ) )
results %>% knitr::kable()



