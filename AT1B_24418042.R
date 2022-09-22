#Load csv Files
library(readr)
df_train <- read.csv("~/Desktop/repurchase_training.csv")

#### EDA ####
#View data
head(df_train)
dim(df_train)


#Install/import packages
library(tidyverse)
library(ggthemes)
library(corrplot)
library(GGally)
library(DT)
library(caret)
library(dlookr)
library(ggplot2)
library(dplyr)




colnames(df_train)

#Search for missing values. No blank cells, many NULL cells.
sapply(df_train, function(x) sum(is.na(x)))


#Check the status of the age of cars
df_train %>%
  count(age_of_vehicle_years) %>%
  ggplot(aes(x=reorder(age_of_vehicle_years, desc(n)), y=n, fill=n)) +
  geom_col() +
  labs(x= "Age of Vehicle (Years)", y= "Count")

##Plot the graph of age distribution of cars in 0 category and 1 category (Target)
#Convert Target variable to factor
df_train$Target <- as.factor(df_train$Target)

##Plot the graph of age distribution of cars in 0 category and 1 category (Target)
ggplot(df_train, aes( x = age_of_vehicle_years,
                      fill = Target)) + geom_bar()

#Check the number of each car model
df_train %>%
  count(car_model) %>%
  ggplot(aes(x=reorder(car_model, desc(n)), y=n, fill=n)) +
  geom_col() +
  labs(x="Car Model", y= "Count")

##Plot the distribution of car model to Target variable 
ggplot(df_train, aes( x = car_model,
                      fill = Target)) + geom_bar()


##Look at the distribution of car model and car segment. 
ggplot(df_train, aes( x = car_model,
                      fill = car_segment))+ geom_bar()

#Compare the age of car and it's model type in relation to target 
ggplot(data=df_train, aes(x=car_model, y= age_of_vehicle_years, fill = Target)) + geom_boxplot()+
  labs(x= "Car Model", y="Age of Vehicle")


#cPlot total services with Target variable
ggplot(df_train, aes( x = total_services,
                      fill = Target))  +  geom_density(alpha = 0.4)

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_var(gender), fill = factor(Target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_var(age_band), fill = factor(Target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_var(car_model), fill = factor(Target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_var(car_segment), fill = factor(Target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_var(as.factor(age_of_vehicle_years)), fill = factor(Target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_var(as.factor(sched_serv_warr)), fill = factor(Target, levels = c("1", "0"))))

#### Linear Classification ####
#Load packages
library(class)
library(naivebayes)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(e1071)
library(gmodels)
library(psych)
library(pROC)
library(caret)
library(datasets)

#### GLM Model ####
df_glm <- df_train
df_glm <- select(df_glm, -age_band, -gender, -ID)

#Create Dummy Variables for car_model and car_segment
dummy_car_model <- dummyVars("~car_model", data=df_glm)
dummy_car_segment <- dummyVars("~car_segment", data = df_glm)
View(df_glm)

#Merge df_glm and dummy variables and prepare df for model. 
dummy_car_model_df <- data.frame(predict(dummy_car_model, newdata= df_glm))
dummy_car_segment_df <- data.frame(predict(dummy_car_segment, newdata= df_glm))
final_df <- data.frame(dummy_car_segment_df, dummy_car_model_df, df_glm)

colnames(final_df)

final_df <- select(final_df, -car_model, -car_segment,-car_modelmodel_19)
View(final_df)

colnames(final_df)

##Dataset has a class imbalance problem. When doing training and test splitting, apply stratified sampling. 
#Research- MLAA Block Session 3, 02_cross_validation.R
set.seed(34)
train.index <- createDataPartition(final_df$Target, p=0.8, list= FALSE)
train <- final_df[train.index,]
train_predictors <- select(train,-Target)
View(train_predictors)
test <- final_df[-train.index,]

##Add more predictors- 
glm_model <- glm(train$Target ~ ., data = train_predictors, family = "binomial")

##Calculate the prob of model
#Do this on train data, then test data
prob_train <- predict(glm_model, train, type = "response")
prob_test <- predict(glm_model, test, type = "response")

#predict using the model
pred_train <- ifelse(prob_train > 0.5, 1, 0)
pred_test <- ifelse(prob_test > 0.5, 1, 0)


View(pred_train)


#Confusion Matrix, F1 and ROC AUC for train set
pred_test <- as.factor(pred_test)
is.factor(pred_test)
train$Target <- as.factor(train$Target)
is.factor(train$Target)

Target_comp <- data.frame()
Target_comp <- data.frame(pred_test, test$Target)

View(Target_comp)

test$Target <- as.factor(test$Target)


#Confusion Matrix, F1 and ROC AUC for test set
confusionMatrix(pred_test, test$Target, mode= "everything")
ROC_test <- roc(test$Target, prob_test)
plot(ROC_test, col="red")
auc(ROC_test)

#Variable Importance plot
vip <- vip(glm_model, bar= FALSE, horizontal = FALSE, size = 1.5)
vip


####Decision Tree####

#Split df_train into training and test sets- stratified sampling
df_tree <- df_train
df_tree <- select(df_tree, -age_band, -gender,-ID)
set.seed(150)
train.index <- createDataPartition(df_tree$Target, p=0.8, list= FALSE)
train <- df_tree[train.index,]
test <- df_tree[-train.index,]

#Basic Decision Tree
model <- rpart(Target ~ .,
               data= train, method = "class", control= rpart.control(cp=0))

#Visualise tree
rpart.plot(model, type=4, box.palette = c("red", "green"), fallen.leaves= TRUE)

#Make prediction
Target_predict <- predict(model, test, type = "class")

#Confusion Matrix and F1 Score
table(Target_predict, test$Target)

is.factor(Target_predict)
is.factor(test$Target)
test$Target <- as.factor(test$Target)
confusionMatrix(Target_predict, test$Target, mode = "everything", positive = "1")

#Compute accuracy
mean(Target_predict == train$Target)

cp <- plotcp(model)


##Improve Decision Tree
#Prune Tree when cp = 0.0011

model2 <- rpart(Target ~ car_model + car_segment + age_of_vehicle_years + total_services,
                data= train, method = "class", control= rpart.control(cp=0.0011))

rpart.plot(model2, type=4, box.palette = c("red", "green"), fallen.leaves= TRUE)

#Make prediction
Target_predict2 <- predict(model2, test, type = "class")

#Confusion Matrix and F1 Score
table(Target_predict2, test$Target)
confusionMatrix(Target_predict2, test$Target, mode = "everything", positive = "1")

is.factor(Target_predict2)
is.factor(test$Target)
test$Target <- as.factor(test$Target)

#Compute accuracy
mean(Target_predict2 == train$Target)

cp <- plotcp(model)

##Partial Dependence Plot
#Research- https://bgreenwell.github.io/pdp/articles/pdp.html
install.packages("pdp")
install.packages("vip")
library(pdp)
library(vip)

#Variable Importance plot
vip <- vip(model, bar= FALSE, horizontal = FALSE, size = 1.5)
View(vip)
vip

vip_dt <- varImp(model, scale= FALSE)
vip_dt



#Single Predictor PDPs
car_age_pdp <- partialPlot(model, pred.data= test, x.var= "age_of_vehicle_years")

#PDP

par.non_sched_serv_warr <- partial(model, pred.var = c("non_sched_serv_warr"), chull = TRUE)
plot.non_sched_serv_warr <- autoplot(par.non_sched_serv_warr, contour= TRUE)
plot.non_sched_serv_warr

par.total_services <- partial(model, pred.var = c("total_services"), chull = TRUE)
plot.total_services <- autoplot(par.total_services, contour= TRUE)
plot.total_services

par.mth_since_last_serv <- partial(model, pred.var = c("mth_since_last_serv"), chull = TRUE)
plot.mth_since_last_serv <- autoplot(par.mth_since_last_serv, contour= TRUE)
plot.mth_since_last_serv

par.total_paid_services <- partial(model, pred.var = c("total_paid_services"), chull = TRUE)
plot.total_paid_services <- autoplot(par.total_paid_services, contour= TRUE)
plot.total_paid_services

par.annualised_mileage <- partial(model, pred.var = c("annualised_mileage"), chull = TRUE)
plot.annualised_mileage <- autoplot(par.annualised_mileage, contour= TRUE)
plot.annualised_mileage

####GLM Validation Test####

val_df <- read.csv("repurchase_validation.csv")
View(val_df)
val_df_predictor <- select(val_df, -age_band, -gender, -ID)


#Merge df_glm and dummy variables and prepare df for model. 
dummy_model_val <- dummyVars("~car_model", data=val_df_predictor)
dummy_segment_val <- dummyVars("~car_segment", data = val_df_predictor)

#Merge df_glm and dummy variables and prepare df for model. 
dummy_model_val <- data.frame(predict(dummy_model_val, newdata= val_df_predictor))
dummy_segment_val <- data.frame(predict(dummy_segment_val, newdata= val_df_predictor))
final_val_df <- data.frame(dummy_model_val, dummy_segment_val, val_df_predictor)

final_val_df <- select(final_val_df,-car_model, -car_segment)

View(final_val_df)

#Use model to predict probabilities and Target 
colnames(train)
colnames(final_val_df)


final_val_df$prob <- predict(glm_model, final_val_df, type= "response")
final_val_df$prediction <- 0

View(final_val_df)


final_val_df$prob
final_val_df[val_df_predictor$prob >= 0.5,"prediction"] <- "1"
View(final_val_df)
final_val_df$prediction

#GLM is over fitting the data and not producing any 1's for Target variable in validation dataset. 
#Decision tree will be used instead pre pruning as the pruning reduced both F1 score and overall accuracy.
  
final_val_df$prediction <-as.factor(val_df_predictor$prediction)
levels(final_val_df$prediction)
final_val_df$prob <- final_val_df$prob[,2]

validation <- data.frame(ID = val_df$ID,
                         target_probability = val_df_predictor$prob,
                         target_class = val_df_predictor$prediction)


####Decision Tree Validation Test####
val_df <- read.csv("~/Desktop/repurchase_validation.csv")
View(val_df)
val_df_predictor <- select(val_df, -age_band, -gender, -ID)



View(val_df)

#Use model to predict probabilities and Target 
colnames(train)
colnames(val_df)

val_df_predictor$prob <- predict(model, val_df_predictor, type= "prob")
val_df_predictor$prediction <- 0
val_df_predictor$prob[,2]
val_df_predictor[val_df_predictor$prob[,2] >= 0.5,"prediction"] <- "1"
View(val_df_predictor)
val_df_predictor$prediction <-as.factor(val_df_predictor$prediction)
levels(val_df_predictor$prediction)
val_df_predictor$prob <- val_df_predictor$prob[,2]

##
validation <- data.frame(ID = val_df$ID,
                         target_probability = val_df_predictor$prob,
                         target_class = val_df_predictor$prediction)


View(validation)
library(readr)
write_csv(validation, file = "repurchase_validation_24418042.csv")
