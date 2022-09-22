#Install and load tidyverse package. 
#Call to read CSV file.
library(readr)
df <- read.csv("~/Desktop/transactions.csv")

#### EDA ####
## Research on what to look for https://www.linkedin.com/pulse/exploratory-data-analysis-linear-regression-oleksii-kharkovyna

#Look at the first 10 rows of the data.
head(df,10)
#Look at the dimensions of the table.
dim(df)
#install "dplyr" and preview data. 
library(dplyr)
glimpse(df)
#Run a summary on each column for all numeric attributes.
summary(df)
#Identify any missing values.
library(skimr)
skim(df)
#Use s.na function.
#No missing data.
mean(is.na(df))
is.na(df)

#The histogram plot is showing positive skewness in industry, location and monthly amount. 
hist(df$monthly_amount,
     xlab = "monthly_amount",
     breaks= sqrt(nrow(df)), xlim = c(0,4000000))


## Outliers 
#Calculate range
max(df$monthly_amount, na.rm= TRUE) - min(df$monthly_amount, na.rm=TRUE)
library(ggplot2)
ggplot(df)+geom_boxplot(aes(monthly_amount))
#There are 6631 observations that are considered outliers. Best to keep them in.

## Visualizations 
# Column graph showing seasonality 
ggplot(df, aes(date, monthly_amount))+geom_col()
#Scatterplot showing increasing amounts towards 2016.
ggplot(df, aes(x=date, y= monthly_amount)) + geom_point()
#Boxplot showing very large range of data and many values which are considered "outliers".
ggplot(df, aes(x=monthly_amount)) + geom_boxplot()
#Scatterplot showing mode in location and industry.
ggplot(df, aes(x=location, y=monthly_amount))+geom_point()
ggplot(df, aes(x=industry, y=monthly_amount))+geom_point()

## Distribution of the variables. 
library(dlookr)
describe(df)
normality(df)
plot_normality(df)
correlate(df)
plot_correlate(df)

#Install packages "ggpubr" and "moments" for creating plots and for computing skewness
library(ggpubr)
library(moments)


#Check distribution of monthly_amount
ggdensity(df, x= "monthly_amount", fill="blue", title= "monthly_amount distribution") +
  stat_overlay_normal_density(color="red", linetype= "dashed")

#Compute Skewness
skewness(df$monthly_amount)
## Transformations over those variables - Log or Square
## Research for how and when to transform data https://www.datanovia.com/en/lessons/transform-data-to-normal-distribution-in-r/
#Perform log transformation of skewed data
df$monthly_amount <- log10(df$monthly_amount+1)
min(df$monthly_amount)

#View transformed data
ggdensity(df, x= "monthly_amount", fill="blue", title= "monthly_amount distribution")+
  stat_overlay_normal_density(color="red", linetype= "dashed")

##Scaling
# summarize the class distribution
percentage <- prop.table(table(df$location)) * 100
cbind(freq=table(df$location), percentage=percentage)

percentage2 <- prop.table(table(df$industry)) * 100
cbind(freq=table(df$industry), percentage=percentage2)
## No need of scaling as both independent variables looks in the same scale

## Converting variables date, location and industry into factors 
#Research https://www.listendata.com/2015/05/converting-multiple-numeric-variables.html#:~:text=In%20R%2C%20you%20can%20convert,be%20set%20as%20factor%20variables.
names <- c('location' ,'industry')
df[,names] <- lapply(df[,names] , factor)
str(df)

#Research https://stackoverflow.com/questions/7439977/changing-date-format-in-r
a<-as.factor(df$date)
abis<-strptime(a,format="%d/%m/%Y") #defining what is the original format of the date
df$date<-as.Date(abis,format="%Y-%m-%d") #defining what is the desired format of the date

## time series Plot 1 of the main data
#Filtering data for Ind = 1 and Loc = 1.
df_filter <- df[df$industry == '1' & df$location == '1',]
View(df_filter)

plot(df_filter$date, df_filter$monthly_amount, type = "l")
ggplot(df_filter, aes(x=date, y=monthly_amount))+geom_line()
colnames(df)

#### Basic Model Fitting ####

## Check for Seasonality.
#Research- https://www.youtube.com/watch?v=wi8vdeiPwuc
library(dplyr)
library(lubridate)
library(forecast)


ts <- ts(df_filter$date, frequency =12, start = c(2013, 01), end = c(2016, 11))
plot(ts)
ts_month <- decompose(ts)
plot(ts_month)




## Creating an aggregated dataset
#Research https://www.datacamp.com/community/tutorials/pipe-r-tutorial
df_agg_mean <- df %>% 
  group_by(date,industry,location) %>% 
  summarise(mean_monthly_amount = mean(monthly_amount))
nrow(df_agg_mean)
View(df_agg_mean)


#Creating dummy variables for Seasonality
#Research- https://www.rdocumentation.org/packages/caret/versions/6.0-91/topics/dummyVars
library(caret)
df_agg_mean$month <- format(df_agg_mean$date, "%m")
colnames(df_agg_mean)
df_agg_mean$month = as.factor(df_agg_mean$month)
is.factor(df_agg_mean$month)

dummy <- dummyVars("~month", data=df_agg_mean)
View(dummy) 
View(df_agg_mean)


final_df <- data.frame(predict(dummy, newdata=df_agg_mean))
View(final_df)


##Merge datasets for train and test split so both sets have dummy variables
new_df <- merge(df_agg_mean,final_df, by.x = 0, by.y = 0)
new_df <- select(new_df,-Row.names,-month.12,-month)
View(new_df)
colnames(new_df)

#Split data into training and test sets. Training set years 2013-2015, test set 2016
#Research- https://www.youtube.com/channel/UCyHEww8_SCdxZvEnkCfi55w

a<-as.factor(df$date)
abis<-strptime(a,format="%d/%m/%Y") #defining what is the original format of the date
df$date<-as.Date(abis,format="%Y-%m-%d") #defining what is the desired format of the date

df$month <-format(df$date,"%m")
colnames(df)

class(df$date)
train_set <- new_df[new_df$date >= "2013-01-01" &
                            new_df$date <= "2015-12-01", ]
test_set <- new_df[new_df$date > "2015-12-01", ]

#Checking split datasets
View(train_set)
View(test_set)
class(train_set$date)
min(train_set$date)
max(train_set$date)

## Filter train and test sets for use later
train_filter <-train_set[train_set$industry == '1' & train_set$location == '1',]
View(train_filter)
colnames(train_filter)

test_filter <- test_set[test_set$industry == '1' & test_set$location == '1',]
View(test_filter)
colnames(test_filter)



# Create linear regression model in train_filter
model_data <- select(train_filter,-industry,-location)
colnames(model_data)
View(model_data)
lm1 <- lm(mean_monthly_amount ~ date +., data = model_data)
summary(lm1)
lm1

#Plot Linear model
library(performance)
check_model(lm1)

plot(lm1)
par(mfrow=c(2, 2))

ggplot(model_data, aes(x=date, y=mean_monthly_amount)) + geom_point() + 
  stat_smooth(method="lm", color="red")

## Creating prediction dataset

date <- as.Date("2016-12-01")
month.01 <- 0           
month.02  <-0
month.03  <-0
month.04  <-0
month.05  <-0
month.06  <-0
month.07  <-0
month.08  <-0
month.09  <-0
month.10  <-0
month.11  <-0

predict_data <- data.frame(date,month.01,month.02,month.03,month.04,month.05,month.06,
                           month.07,month.08,month.09,month.10,month.11)

a<-as.factor(predict_data$date)
abis<-strptime(a,format="%d/%m/%Y") #defining what is the original format of the date
predict_data$date<-as.Date(abis,format="%Y-%m-%d") #defining what is the desired format of the date

predict_data$date<-as.Date(abis,format="%Y-%m-%d")
class(predict_data$date)
colnames(model_data)


colnames(predict_data)
class(predict_data$date)



## Prediction
summary(lm1)
prediction <- predict(lm1, newdata = predict_data)
prediction
AIC(lm1)
#Reverse log transformation
10**5.099136
View(predict_data)
View(model_data)


ggplot(test_filter, aes(x=date, y=mean_monthly_amount)) + geom_point() +
  geom_smooth(prediction_test, color= "red")

##Calcuate Mean Squared Error
mean(summary(lm1)$residuals^2)

##Apply model to test set for loc 1 & ind 1
test_filter <- select(test_filter,-industry,-location)
prediction_test <- predict(lm1, newdata = test_filter)
View(prediction_test)
dec_test_pred <- forecast(prediction_test, level=95, h=1)
dec_test_pred
10**5.115415


#### Applying model to all industries and locations ####
library(data.table)
industry_location <- copy(new_df)
View(industry_location)

output <- data.frame()

# determine what we will be looping over
industries = unique(industry_location$industry)
locations = unique(industry_location$location)
industries
locations
#colnames(industry_location)
#colnames(industry_location[,c(1,4,5:15)])

#start loop code
#Reference- hint for AT1 used, published on Canvas.

for (ind in industries) {
  
  for (loc in locations) {
    
    # create a subset of the data
    temp = industry_location[industry_location$industry == ind &
                               industry_location$location == loc, ]
    
    # Check to make sure you have at least X months of data
    if (length(unique(temp$date)) >= 36) {
      
      # train your model
      # INSERT YOUR MODEL TRAINING CODE, INCLUDING TRAINING/TEST PARTITIONING
      temp_train_set <- temp[temp$date >= "2013-01-01" &
                               temp$date <= "2015-12-01", ]
      temp_test_set <- temp[temp$date > "2015-12-01", ]
      
      
      model <- lm(mean_monthly_amount ~ date +., data = temp_train_set[,c(1,4,5:15)])
      # output a prediction
      temp$prediction = predict(model, temp[,c(1,4,5:15)])
      
      # CALCULATE YOUR ERROR
      
      error <- mean(summary(model)$residuals^2)
      prediction <- predict(model, newdata = predict_data)
      
      # append your error to the output data frame, include industry and location variable
      
      output = rbind(output, error)
      prediction_data = rbind(prediction)
      output[nrow(output),2]<- ind
      output[nrow(output),3]<- loc
      output[nrow(output),4]<- prediction
      
    }
  }
  
  
}
# end loop code
## Renaming column names in the output dataframe
names(output)[1] <- "error"
names(output)[2] <- "industry"
names(output)[3] <- "location"
names(output)[4] <- "Dec_2016_prediction_log10"
output$actual_predction_Dec_2016 <- 10**(output$Dec_2016_prediction_log10)
View(output)

##Export output data
library(readr)
write_csv(output, file = "output.csv")

##Plots to view worst performing industry and location
plot(error ~ industry, col="blue", pch=19, cex=2, data=output)
text(error ~ industry, labels=error, data=output, cex=0.9, font=2)

plot(error ~ location, col="red", pch=19, cex=2, data=output)



*******************************************************************************************
  
  ### Calculating error for the test data
output_test_error <- data.frame()

  for (ind in industries) {
    
    for (loc in locations) {
      
      # create a subset of the data
      temp_test = industry_location[industry_location$industry == ind &
                                 industry_location$location == loc, ]
      
      # Check to make sure you have at least X months of data
      if (length(unique(temp$date)) >= 36) {
        
        # train your model
        # INSERT YOUR MODEL TRAINING CODE, INCLUDING TRAINING/TEST PARTITIONING
        temp_train_set <- temp[temp$date >= "2013-01-01" &
                                 temp$date <= "2015-12-01", ]
        temp_test_set <- temp[temp$date > "2015-12-01", ]
        
        
        model <- lm(mean_monthly_amount ~ date +., data = temp_train_set[,c(1,4,5:15)])
        # output a prediction
        temp_test$prediction_test = predict(model, temp_test_set[,c(1,4,5:15)])
        
        # CALCULATE YOUR ERROR
        
        error_test <- mean(summary(model)$residuals^2)
        prediction_test <- predict(model, newdata = predict_data)
        
        # output$industry <-ind
        #output$location <- loc
        
        #output <-cbind(output,ind,loc)
        
        # append your error to the output data frame, include industry and location variable
        
        
        output_test_error = rbind(output_test_error, error_test)
        prediction_data_test = rbind(prediction_test)
        output_test_error[nrow(output),2]<- ind
        output_test_error[nrow(output),3]<- loc
        output_test_error[nrow(output),4]<- prediction_test
      }
    }
    
    
  }
# end loop code
## Renaming column names in the output dataframe
names(output_test_error)[1] <- "error"
names(output_test_error)[2] <- "industry"
names(output_test_error)[3] <- "location"
names(output_test_error)[4] <- "Dec_2016_prediction_log10"

output_test_error$actual_predction_Dec_2016 <- 10**(output_test_error$Dec_2016_prediction_log10)
View(output_test_error)


