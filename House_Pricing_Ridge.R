install.packages("mlbench")
install.packages("psych")
install.packages("glmnet")
install.packages("tidyverse")
install.packages("lmtest")
install.packages("caret")
install.packages("Hmisc")
install.packages("corrplot")
install.packages("dplyr")
install.packages("PerformanceAnalytics")      #Correlation Chart Matrix
library(mlbench)
library(psych)
library(glmnet)
library(tidyverse)
library(lmtest)
library(caret)
liblibrary(corrplot)
lirary(Hmisc)
library(dplyr)
library(PerformanceAnalytics)

#IMPORTING THE DATASET
data=read.csv("E:\\Pavi\\LOYOLA PG\\Applied Regression Analysis\\Project\\Boston House\\boston.csv")
data=select(data,-c(1))
head(data)
dim(data)

y=c(data$CRIM,data$ZN,data$CHAS,data$NOX,data$RM,data$AGE,data$DIS,data$RAD,data$TAX,data$PTRATIO,data$B,data$LSTAT)

xyplot(data$MEDV~y,type=c("p","smooth"))

par(mfrow=c(5,3))

boxplot(data$CRIM,col=50,xlab="CRIM")
boxplot(data$ZN,col=2,xlab="ZN")
boxplot(data$CHAS,col="BLUE",xlab="CHAS")
boxplot(data$NOX,col=4,xlab="NOX")
boxplot(data$RM,col=5,xlab="RM")
boxplot(data$AGE,col=6,xlab="AGE")
boxplot(data$DIS,col=7,xlab="DIS")
boxplot(data$RAD,col=8,xlab="RAD")
boxplot(data$TAX,col="YELLOW",xlab="TAX")
boxplot(data$PTRATIO,col=10,xlab="PTRATIO")
boxplot(data$B,col=11,xlab="B")
boxplot(data$LSTAT,col=43,xlab="LSTAT")
boxplot(data$MEDV,col=15,xlab="MEDV")

#LINEARITY BETWEEN STUDY VARIABLE AND EXPLANATORY VARIABLES
xyplot(data$MEDV~data$CRIM,type=c("p","smooth"),pch=16,col=2,main="Relationship Between Y and X1") 
xyplot(data$MEDV~data$ZN,type=c("p","smooth"),col=3,main="Relationship Between Y and X2")
xyplot(data$MEDV~data$INDUS,type=c("p","smooth"),col=4,main="Relationship Between Y and X3")
xyplot(data$MEDV~data$CHAS,type=c("p","smooth"),col=5,main="Relationship Between Y and X4")
xyplot(data$MEDV~data$NOX,type=c("p","smooth"),col=5,main="Relationship Between Y and X5")
xyplot(data$MEDV~data$RM,type=c("p","smooth"),col=6,main="Relationship Between Y and X6")
xyplot(data$MEDV~data$AGE,type=c("p","smooth"),col=7,main="Relationship Between Y and X7")
xyplot(data$MEDV~data$DIS,type=c("p","smooth"),col=2,main="Relationship Between Y and X8")
xyplot(data$MEDV~data$RAD,type=c("p","smooth"),col=24,main="Relationship Between Y and X9")
xyplot(data$MEDV~data$TAX,type=c("p","smooth"),col=22,main="Relationship Between Y and X10")
xyplot(data$MEDV~data$PTRATIO,type=c("p","smooth"),col=12,main="Relationship Between Y and X11")
xyplot(data$MEDV~data$B,type=c("p","smooth"),col=35,main="Relationship Between Y and X12")
xyplot(data$MEDV~data$LSTAT,type=c("p","smooth"),col=28,main="Relationship Between Y and X13")

sjp.lmer(model_mlrm)

plot(data$MEDV,data$CRIM,pch=16)

outliersper <- function(x)
{
  length(which(x > mean(x) + 3 * sd(x) | x < mean(x) - 3 * sd(x)) ) / length(x)
}

outliersper(data$MEDV)
outliersper(data$CRIM)

#REMOVAL OF OUTLIERS in MEDV
q1=quantile(data$MEDV,0.25)
q3=quantile(data$MEDV,0.75)
iqr=IQR(data$MEDV)

data1=subset(data,data$MEDV>(q1-(1.5*iqr)) & data$MEDV<(q1+(1.5*iqr)))


#_______________________________________________________________________________________________________________________________________________

#MLRM
model_mlrm=lm(MEDV~.,data)
summary(model_mlrm)

error=residuals(model_mlrm)
n=length(error)
MSRes=sum(error^2)/n-14

#Standardized Residuals
data1=data
data1$di=error/sqrt(MSRes)
head(data1)

#Removing Outliers
data1=subset(data1,di<=3)

model_mlrm1=lm(MEDV~.,data1[,-15])
summary(model_mlrm1)

#VIOLATION OF ASSUMPTIONS

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
plot(fitted(model_mlrm),residuals(model_mlrm))
abline(0,0)

plot(fitted(model_mlrm1),residuals(model_mlrm1))
abline(0,0)


#QQ-PLOT of RESIDUALS
qqnorm(residuals(model_mlrm))
qqline(residuals(model_mlrm))

qqnorm(residuals(model_mlrm1))
qqline(residuals(model_mlrm1))

#NORMALITY OF ERRORS/RESIDUALS
plot(density(residuals(model_mlrm)))
plot(density(residuals(model_mlrm1)))

#_______________________________________________________________________________________________________________________________________________

#MLRM - Step Wise MLRM
model_mlrm_step=step(lm(MEDV~.,data))
summary(model_mlrm_step)

data2=subset(data,abs(rstandard(model_mlrm_step))<1.96)
model_mlrm_step1=step(lm(MEDV~.,data2))
summary(model_mlrm_step1)

#-------------------------------------------------------------

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
plot(fitted(model_mlrm_step),residuals(model_mlrm_step))
abline(0,0)

plot(fitted(model_mlrm_step1),residuals(model_mlrm_step1))
abline(0,0)


#QQ-PLOT of RESIDUALS
qqnorm(residuals(model_mlrm_step))
qqline(residuals(model_mlrm_step))

qqnorm(residuals(model_mlrm_step1))
qqline(residuals(model_mlrm_step1))

#NORMALITY OF ERRORS/RESIDUALS
plot(density(residuals(model_mlrm_step)))
plot(density(residuals(model_mlrm_step1)))

#_______________________________________________________________________________________________________________________________________________

### SPLITTING THE DATA INTO TRAIN AND TEST ###

install.packages("caTools")
library(caTools)
set.seed(25)

sample_size=floor(0.7*nrow(data))
train_ind=sample(seq_len(nrow(data)),size=sample_size)
train=data[train_ind,]  #data[r,c]
test=data[-train_ind,]
dim(train)
dim(test)

#----------------------------------------

x_train=train[,1:13]
x_test=test[,1:13]
y_train=train[,14]
y_test=test[,14]

#_______________________________________________________________________________________________________________________________________________

#MLRM - TRAIN
model_mlrm_train=lm(MEDV~.,train)
summary(model_mlrm_train)

#Removal of Outliers
data3=subset(train,abs(rstandard(model_mlrm_train))<1.96)
model_mlrm_train1=lm(MEDV~.,data3)
summary(model_mlrm_train1)

boxplot(residuals(model_mlrm_train),col=15,xlab="MEDV")
boxplot(residuals(model_mlrm_train1),col=15,xlab="MEDV")



#-----------------------------------------------------

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(model_mlrm_train),residuals(model_mlrm_train),main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(model_mlrm_train1),residuals(model_mlrm_train1),main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)


#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(model_mlrm_train),main="QQ Plot With Outliers")
qqline(residuals(model_mlrm_train))

qqnorm(residuals(model_mlrm_train1),main="QQ Plot Without Outliers")
qqline(residuals(model_mlrm_train1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(model_mlrm_train)),main="Density Plot with Outliers",col=3)
plot(density(residuals(model_mlrm_train1)),main="Density Plot without Outliers",col=5)

#AUTOCORRELATION VISAULIZATION
plot(residuals(model_mlrm_train1),col=2,main="Detection of Autocorrelation for MLRM")

#------------------------------------------

#PREDICTING MEDV-TEST using TRAIN
pred=predict(model_mlrm_train1,test)
R2=summary(model_mlrm_train)$r.squared
R2_without_outliers=summary(model_mlrm_train1)$r.squared
SSR=sum((test$MEDV-pred)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_test_mlrm=1-(SSR/SST)
RMSE_test_mlrm=sqrt((SSR)/nrow(test))
R2
R2_without_outliers
R_square_test_mlrm
RMSE_test_mlrm

#plot(pred,test$MEDV-pred)
#abline(0,0)

#qqnorm(test$MEDV-pred)
#qqline(test$MEDV-pred)

#plot(density(test$MEDV-pred))



#_______________________________________________________________________________________________________________________________________________


#MULTICOLLINEARITY TEST
install.packages("mctest")
library(mctest)
omcdiag(model_mlrm1)
imcdiag(model_mlrm1)
omcdiag(model_mlrm_step1)
imcdiag(model_mlrm_step1)
omcdiag(model_poly1)
imcdiag(model_poly1)
omcdiag(polyridge1)
imcdiag(polyridge1)
omcdiag(polylasso1)
imcdiag(polylasso1)
omcdiag(polyelast_net1)
imcdiag(polyelast_net1)


model_collinearity=lm(MEDV~CRIM+ZN+INDUS+CHAS+NOX+RM+AGE+DIS+PTRATIO+B+LSTAT,data)
summary(model_collinearity)

#_______________________________________________________________________________________________________________________________________________

#MLRM
model_mlrm_test=lm(log(MEDV)~.,test)
summary(model_mlrm_test)

#MLRM - Step Wise MLRM
model_mlrm_step=step(lm(MEDV~.,train))
summary(model_mlrm_step)

#Removal of Outliers
data4=subset(train,abs(rstandard(model_mlrm_step))<1.96)
model_mlrm_step1=step(lm(MEDV~.,data4))
summary(model_mlrm_step1)

#------------------------------------------

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(model_mlrm_step),residuals(model_mlrm_step),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(model_mlrm_step1),residuals(model_mlrm_step1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)


#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(model_mlrm_step),col=9,main="QQ Plot With Outliers")
qqline(residuals(model_mlrm_step))

qqnorm(residuals(model_mlrm_step1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(model_mlrm_step1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(model_mlrm_step)),main="Density Plot with Outliers",col=3)
plot(density(residuals(model_mlrm_step1)),main="Density Plot without Outliers",col=5)

#AUTOCORRELATION VISAULIZATION
plot(residuals(model_mlrm_step1),col=3,main="Detection of Autocorrelation for MLRM - STEP WISE")

#------------------------------------------

#PREDICTING MEDV-TEST using TRAIN
pred_step=predict(model_mlrm_step1,test)
R2=summary(model_mlrm_step)$r.squared
R2_without_outliers=summary(model_mlrm_step1)$r.squared
SSR=sum((test$MEDV-pred_step)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_test_step=1-(SSR/SST)
RMSE_test_step=sqrt((SSR)/nrow(test))
R2
R2_without_outliers
R_square_test_step
RMSE_test_step


#_______________________________________________________________________________________________________________________________________________

#POLYNOMIAL WITH 2nd DEGREE
install.packages("tidyverse")
library(tidyverse)
theme_set(theme_classic())

#model_poly=lm(MEDV~poly(LSTAT,5,raw=TRUE),data=train)
model_poly=lm(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),data=train)
summary(model_poly)

#Removal of Outliers
data5=subset(train,abs(rstandard(model_poly))<1.96)
model_poly1=lm(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),data=data5)
summary(model_poly1)

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(model_poly),residuals(model_poly),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(model_poly1),residuals(model_poly1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)


#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(model_poly),col=9,main="QQ Plot With Outliers")
qqline(residuals(model_poly))

qqnorm(residuals(model_poly1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(model_poly1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(model_poly)),main="Density Plot with Outliers",col=3)
plot(density(residuals(model_poly1)),main="Density Plot without Outliers",col=5)

#AUTOCORRELATION VISAULIZATION
par(mfrow=c(1,2))

plot(residuals(model_poly),col=3,main="Detection of Autocorrelation for Polynomial - With Outliers")
plot(residuals(model_poly1),col=3,main="Detection of Autocorrelation for Polynomial - Without Outliers")


#------------------------------------------

#PREDICTING MEDV-TEST using TRAIN
pred_poly=predict(model_poly1,test)
R2=summary(model_poly)$r.squared
R2_without_outliers=summary(model_poly1)$r.squared
SSR=sum((test$MEDV-pred_poly)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_test_poly=1-(SSR/SST)
RMSE_poly=sqrt((SSR)/nrow(test))
R2
R2_without_outliers
R_square_test_poly
RMSE_poly

#_______________________________________________________________________________________________________________________________________________

#RIDGE REGRESSION
#install.packages("caret")
custom<- trainControl(method = "repeatedcv",number=10,repeats=5,verboseIter= T)

set.seed(123)
ridge<-train(MEDV~.,data= train,method='glmnet',
             tunegrid=expand.grid(alpha=0,lambda=seq(0.0001,1,length=5)),
             trControl=custom)
ridge

#MODEL AFTER REMOVING OUTLIERS:

error=residuals(ridge)
n=length(error)
MSRes= sum(error^2)/(n-14)
data9=subset(train,(error/sqrt(MSRes))<3)
ridge1<-train(MEDV~.,
               data= data9,
               method='glmnet',
               tunegrid=expand.grid(alpha=0,
                                  lambda=seq(0.0001,1,length=5)),
               trControl=custom)
ridge1

#MODEL BUILDING USING BEST LAMBDA
ridge2=glmnet(x_train,y_train,alpha = 0.1,lambda = 0.1386975)
coef(ridge2)

#plotting results
plot(ridge1)
plot(ridge1$finalModel,xvar="lambda",label=T)
plot(ridge1$finalModel,xvar="dev",label=T)
plot(varImp(ridge1,scale=F))


#PREDICTING MEDV-TEST using TRAIN
pred_ridge1=predict(ridge1,test)
SSR=sum((test$MEDV-pred_ridge1)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_ridge_test1=1-(SSR/SST)
RMSE_ridge1=sqrt((SSR)/nrow(test))
R_square_ridge_test1
RMSE_ridge1

#AUTOCORRELATION
plot(residuals(ridge1),col="green",main="error in ridge")

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(ridge),residuals(ridge),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(ridge1),residuals(ridge1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)

#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(ridge),col=9,main="QQ Plot With Outliers")
qqline(residuals(ridge))

qqnorm(residuals(ridge1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(ridge1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(ridge)),main="Density Plot with Outliers",col=3)
plot(density(residuals(ridge1)),main="Density Plot without Outliers",col=5)

#------------------------------------------

#-------------------------------------------------------------------------------

#LASSO REGRESSION

set.seed(123)
lasso<-train(MEDV ~ .,
             train,
             method='glmnet',
             tuneGrid= expand.grid(alpha=1,lambda=seq(0.001,0.2,length=5)),
             trControl=custom)
lasso

#MODEL AFTER REMOVING OUTLIERS:

error=residuals(lasso)
n=length(error)
MSRes= sum(error^2)/(n-14)
data10=subset(train,(error/sqrt(MSRes))<3)
lasso1<-train (MEDV~.,
               data= data10,
               method='glmnet',
               tunegrid=expand.grid(alpha=1,
                                  lambda=seq(0.0001,1,length=5)),
               trControl=custom)
lasso1

#MODEL BUILDING USING BEST LAMBDA
lasso2=glmnet(x_train,y_train,alpha = 0.1,lambda = 0.1386975)
coef(lasso2)

plot(lasso1)
plot(lasso1$finalModel,xvar='lambda',label=T)
plot(lasso1$finalModel,xvar='dev',label=T)
plot(varImp(lasso1,scale=F))

#PREDICTING MEDV USING TEST DATA
pred_lasso1=predict(lasso1,test)
head(pred_lasso1)
SSR=sum((test$MEDV-pred_lasso1)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_test_lasso1=1-(SSR/SST)
R_square_test_lasso1
RMSE_test_lasso1=sqrt((SSR)/nrow(test))
RMSE_test_lasso1
pred_lasso1

#AUTOCORRELATION
plot(residuals(lasso1),col="red",main="error in lasso")


#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(lasso),residuals(lasso),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(lasso1),residuals(lasso1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)

#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(lasso),col=9,main="QQ Plot With Outliers")
qqline(residuals(lasso))

qqnorm(residuals(lasso1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(lasso1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(lasso)),main="Density Plot with Outliers",col=3)
plot(density(residuals(lasso1)),main="Density Plot without Outliers",col=5)


#----------------------------------------------------------------------------------

#Elastic Net Regression
set.seed(123)
elast_net<- train(MEDV ~.,
                  train,
                  method='glmnet',
                  tuneGrid=expand.grid(alpha=seq(0,1,length=10),
                                       lambda=seq(0.0001,1,length=5)),
                  trControl=custom)
elast_net

#MODEL AFTER REMOVING OUTLIERS:

error=residuals(elast_net)
n=length(error)
MSRes= sum(error^2)/(n-14)
data11=subset(train,(error/sqrt(MSRes))<3)
elast_net1<-train (MEDV~.,
               data= data11,
               method='glmnet',
               tunegrid=expand.grid(alpha=0,
                                  lambda=seq(0.0001,1,length=5)),
               trControl=custom)
elast_net1

#MODEL BUILDING USING BEST LAMBDA
elast_net2=glmnet(x_train,y_train,alpha = 0.1,lambda =0.1386975)
coef(elast_net2)

plot(elast_net1)
plot(elast_net1$finalModel, xvar='lambda',label=T)
(elast_net1$finalModel, xvar='dev',label=T)
plot(varImp(elast_net1))

#PREDICTING MEDV USING TEST DATA

pred_elastic1=predict(elast_net1,test)
head(ppred_elastic1red)
SSR=sum((test$MEDV-pred_elastic1)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_test_elast1=1-(SSR/SST)
R_square_test_elast1
RMSE_test_elast1=sqrt((SSR)/nrow(test))
RMSE_test_elast1
pred_elastic1

#AUTOCORRELATION
plot(residuals(elast_net1),col="blue",main="error in elasticnet")


#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(elast_net),residuals(elast_net),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(elast_net1),residuals(elast_net1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)

#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(elast_net),col=9,main="QQ Plot With Outliers")
qqline(residuals(elast_net))

qqnorm(residuals(elast_net1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(elast_net1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(elast_net)),main="Density Plot with Outliers",col=3)
plot(density(residuals(elast_net1)),main="Density Plot without Outliers",col=5)


#----------------------------------------------------------------------------

#Compare models

model_list <- list(Ridge=ridge1,Lasso=lasso1,ElasticNet=elast_net1)
res<-resamples(model_list)
summary(res)


#_______________________________________________________________________________________________________________________________________________

#RIDGE USING POLYNOMIAL 2nd DEGREE

#custom control parameters

custom<- trainControl(method = "repeatedcv",number=10,repeats=5,verboseIter= T)
#ridge regression
set.seed(123)

#MODEL BUILDING
polyridge<-train(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),
                 data=train,method='glmnet',
                 tunegrid=expand.grid(alpha=0,lambda=seq(0.0001,1,length=5)),
                  trControl=custom)

#Removal of Outliers
error=residuals(polyridge)
n=length(error)
MSRes=sum(error^2)/(n-14)
data6=subset(train,(error/sqrt(MSRes))<3)
polyridge1<-train(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),
                  data=data6,method='glmnet',
                  tunegrid=expand.grid(alpha=0,lambda=seq(0.0001,1,length=5)),
                  trControl=custom)

#MODEL BUILDING USING BEST LAMBDA
polyridge2=glmnet(x_train,y_train,alpha=0.1,lambda=0.0144)
coef(polyridge2)

mean(polyridge1$resample$RMSE)

#PREDICTING MEDV-TEST using TRAIN
pred_ridge=predict(polyridge1,test)
SSR=sum((test$MEDV-pred_ridge)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_ridge_test=1-(SSR/SST)
RMSE_ridge=sqrt((SSR)/nrow(test))
R_square_ridge_test
RMSE_ridge

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(polyridge),residuals(polyridge),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(polyridge1),residuals(polyridge1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)

#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(polyridge),col=9,main="QQ Plot With Outliers")
qqline(residuals(polyridge))

qqnorm(residuals(polyridge1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(polyridge1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(polyridge)),main="Density Plot with Outliers",col=3)
plot(density(residuals(polyridge1)),main="Density Plot without Outliers",col=5)

#AUTOCORRELATION VISAULIZATION
par(mfrow=c(1,2))


plot(residuals(polyridge),col=5,main="Detection of Autocorrelation for Poly Ridge - With Outliers")
plot(residuals(polyridge1),col=3,main="Detection of Autocorrelation for Poly Ridge - Without Outliers")

#------------------------------------------

#plotting results
plot(polyridge1)
polyridge1
plot(polyridge1$finalModel,xvar="lambda",label=T)
plot(polyridge1$finalModel,xvar="dev",label=T)
plot(varImp(polyridge1,scale=F))

#_______________________________________________________________________________________________________________________________________________

#POLY LASSO REGRESSION

set.seed(123)
polylasso<-train(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),
                 data=train,method='glmnet',
                 tuneGrid= expand.grid(alpha=1,lambda=seq(0.001,0.2,length=5)),
                 trControl=custom)

#Removal of Outliers
error=residuals(polylasso)
n=length(error)
MSRes=sum(error^2)/(n-14)
data7=subset(train,(error/sqrt(MSRes))<3)
polylasso1<-train(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),
                 data=data7,method='glmnet',
                 tuneGrid= expand.grid(alpha=1,lambda=seq(0.001,0.2,length=5)),
                 trControl=custom)

#MODEL BUILDING USING BEST LAMBDA
polylasso2=glmnet(x_train,y_train,alpha=1,lambda=0.001)
coef(polylasso2)


#TEST DATA
pred_lasso=predict(polylasso1,test)
head(pred_lasso)
SSR=sum((test$MEDV-pred_lasso)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_lasso=1-(SSR/SST)
RMSE_lasso=sqrt((SSR)/nrow(test))
R_square_lasso
RMSE_lasso

#------------------------------------------------------------

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(polylasso),residuals(polylasso),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(polylasso1),residuals(polylasso1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)

#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(polylasso),col=9,main="QQ Plot With Outliers")
qqline(residuals(polylasso))

qqnorm(residuals(polylasso1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(polylasso1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(polylasso)),main="Density Plot with Outliers",col=3)
plot(density(residuals(polylasso1)),main="Density Plot without Outliers",col=5)


#AUTOCORRELATION VISAULIZATION
par(mfrow=c(1,2))

plot(residuals(polylasso),col=4,main="Detection of Autocorrelation for Poly Lasso - With Outliers")
plot(residuals(polylasso1),col=3,main="Detection of Autocorrelation for Poly Lasso - Without Outliers")

#------------------------------------------


#Plot results
plot(polylasso1)
polylasso1
plot(polylasso1$finalModel,xvar='lambda',label=T)
plot(polylasso1$finalModel,xvar='dev',label=T)
plot(varImp(polylasso1,scale=F))

#_______________________________________________________________________________________________________________________________________________

#ELASTIC NET REGRESSION

set.seed(123)
polyelast_net<- train(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),
                     data=train,method='glmnet',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),
                     lambda=seq(0.0001,1,length=5)),trControl=custom)

#Removal of Outliers
error=residuals(polyelast_net)
n=length(error)
MSRes=sum(error^2)/(n-14)
data8=subset(train,(error/sqrt(MSRes))<3)
polyelast_net1<- train(MEDV~poly(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,degree=2,raw=TRUE),
                     data=data8,method='glmnet',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),
                     lambda=seq(0.0001,1,length=5)),trControl=custom)

#MODEL BUILDING USING BEST LAMBDA
polyelast_net2=glmnet(x_train,y_train,alpha=0.111,lambda=0.0001)
coef(polyelast_net2)

#TEST DATA
pred_elastic=predict(polyelast_net1,test)
head(pred_elastic)
SSR=sum((test$MEDV-pred_elastic)^2)
SST=sum((test$MEDV-mean(test$MEDV))^2)
R_square_elastic=1-(SSR/SST)
RMSE_elastic=sqrt((SSR)/nrow(test))
R_square_elastic
RMSE_elastic


#------------------------------------------------------------

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
par(mfrow=c(1,2))
plot(fitted(polyelast_net),residuals(polyelast_net),col=2,main="Error vs Predicted MEDV - With Outliers")
abline(0,0)

plot(fitted(polyelast_net1),residuals(polyelast_net1),col=5,main="Error vs Predicted MEDV - Without Outliers")
abline(0,0)

#QQ-PLOT of RESIDUALS
par(mfrow=c(1,2))

qqnorm(residuals(polyelast_net),col=9,main="QQ Plot With Outliers")
qqline(residuals(polyelast_net))

qqnorm(residuals(polyelast_net1),col=8,main="QQ Plot Without Outliers")
qqline(residuals(polyelast_net1))

#NORMALITY OF ERRORS/RESIDUALS
par(mfrow=c(1,2))

plot(density(residuals(polyelast_net)),main="Density Plot with Outliers",col=3)
plot(density(residuals(polyelast_net1)),main="Density Plot without Outliers",col=5)

#AUTOCORRELATION VISAULIZATION
par(mfrow=c(1,2))

plot(residuals(polyelast_net),col=4,main="Detection of Autocorrelation for Poly Elastic - With Outliers")
plot(residuals(polyelast_net1),col=3,main="Detection of Autocorrelation for Poly Elastic - Without Outliers")

#------------------------------------------

#Plot results

plot(polyelast_net1)
polyelast_net1
plot(polyelast_net1$finalModel, xvar='lambda',label=T)
plot(polyelast_net1$finalModel, xvar='dev',label=T)
plot(varImp(polyelast_net1))

#Compare models
model_list <- list(Ridge=ridge1,Lasso=lasso1,ElasticNet=elast_net1)
res<-resamples(model_list)
summary(res)


#COMPARING MODELS - RIDGE vs LASSO vs ELASTIC NET
model_list <- list(Ridge=ridge1,Lasso=lasso1,ElasticNet=elast_net1,Poly_Ridge=polyridge1,Poly_Lasso=polylasso1,Poly_ElasticNet=polyelast_net1)
res<-resamples(model_list)
summary(res)



#CONCLUSION-TRAIN
#R-Squared Value
R_MLRM=summary(model_mlrm1)$r.squared
R_MLRM_STEP=summary(model_mlrm_step1)$r.squared
R_POLY=summary(model_poly1)$r.squared
R_RIDGE=mean(ridge1$resample$Rsquared)
R_LASSO=mean(lasso1$resample$Rsquared)
R_ELASTIC=mean(elast_net1$resample$Rsquared)
R_POLY_RIDGE=mean(polyridge1$resample$Rsquared)
R_POLY_LASSO=mean(polylasso1$resample$Rsquared)
R_POLY_ELASTIC=mean(polyelast_net1$resample$Rsquared)

#Adjusted R-Squared Value
ADJ_R_MLRM=summary(model_mlrm1)$adj.r.squared
ADJ_R_MLRM_STEP=summary(model_mlrm_step1)$adj.r.squared
ADJ_R_POLY=summary(model_poly1)$adj.r.squared
ADJ_R_RIDGE=''
ADJ_R_LASSO=''
ADJ_R_ELASTIC=''
ADJ_R_POLY_RIDGE=''
ADJ_R_POLY_LASSO=''
ADJ_R_POLY_ELASTIC=''


#RMSE Value
RMSE_MLRM=sigma(model_mlrm1)
RMSE_MLRM_STEP=sigma(model_mlrm_step1)
RMSE_POLY=sigma(model_poly1)
RMSE_RIDGE=mean(ridge1$resample$RMSE)
RMSE_LASSO=mean(lasso1$resample$RMSE)
RMSE_ELASTIC=mean(elast_net1$resample$RMSE)
RMSE_POLY_RIDGE=mean(polyridge1$resample$RMSE)
RMSE_POLY_LASSO=mean(polylasso$resample$RMSE)
RMSE_POLY_ELASTIC=mean(polyelast_net1$resample$RMSE)

#F-Statistic Value
F_MLRM=summary(model_mlrm1)$fstatistic[1]
F_MLRM_STEP=summary(model_mlrm_step1)$fstatistic[1]
F_POLY=summary(model_poly1)$fstatistic[1]
F_RIDGE=''
F_LASSO=''
F_ELASTIC=''
F_POLY_RIDGE=''
F_POLY_LASSO=''
F_POLY_ELASTIC=''

Model=c("MLRM","MLRM - Step Wise", "Polynomial", "Ridge", "Lasso", "Elastic-Net","Poly-Ridge", "Poly-Lasso","Poly-Elastic Net")
R_Square = c(R_MLRM,R_MLRM_STEP,R_POLY,R_RIDGE,R_LASSO,R_ELASTIC,R_POLY_RIDGE,R_POLY_LASSO,R_POLY_ELASTIC)
Adj_R_Square = c(ADJ_R_MLRM,ADJ_R_MLRM_STEP,ADJ_R_POLY,ADJ_R_RIDGE,ADJ_R_LASSO,ADJ_R_ELASTIC,ADJ_R_POLY_RIDGE,ADJ_R_POLY_LASSO,ADJ_R_POLY_ELASTIC)
RMSE = c(RMSE_MLRM,RMSE_MLRM_STEP,RMSE_POLY,RMSE_RIDGE,RMSE_LASSO,RMSE_ELASTIC,RMSE_POLY_RIDGE,RMSE_POLY_LASSO,RMSE_POLY_ELASTIC)
F_Statistic=c(F_MLRM,F_MLRM_STEP,F_POLY,F_RIDGE,F_LASSO,F_ELASTIC,F_POLY_RIDGE,F_POLY_LASSO,F_POLY_ELASTIC)

#Table-TRAIN
train_table=data.frame(Model,R_Square,Adj_R_Square,RMSE,F_Statistic)
comparison_train=list()
comparison_train[[1]] = "Comparison of Different Models for Train Data"
comparison_train[[2]]=train_table 
comparison_train

#-------------------------------------------

#CONCLUSION - TEST DATA

Model_Test=c("MLRM","MLRM - Step Wise", "Polynomial", "Ridge", "Lasso", "Elastic-Net","Poly-Ridge", "Poly-Lasso","Poly-Elastic Net")
R_Square_test= c(R_square_test_mlrm,R_square_test_step,R_square_test_poly,R_square_ridge_test1,
                 R_square_test_lasso1,R_square_test_elast1,R_square_ridge_test,R_square_lasso,R_square_elastic)
RMSE_test= c(RMSE_test_mlrm,RMSE_test_step,RMSE_poly,RMSE_ridge1,RMSE_test_lasso1,RMSE_test_elast1,RMSE_ridge,RMSE_lasso,RMSE_elastic)


#Table-TRAIN
test_table=data.frame(Model_Test,R_Square_test,RMSE_test)
comparison_test=list()
comparison_test[[1]] = "Comparison of Different Models for Test Data"
comparison_test[[2]]=test_table 
comparison_test

#COMPARING PREDICTED vs ACTUAL VALUES
par(mfrow=c(3,2))

plot(test$MEDV,pch=10,col="red",main="MULTIPLE LINEAR REGRESSION MODEL")
lines(pred,col="blue")

plot(test$MEDV,pch=10,col="black",main="MULTIPLE LINEAR REGRESSION MODEL - STEP METHOD")
lines(pred_step,col="pink")

plot(test$MEDV,pch=10,col="blue",main="POLYNOMIAL REGRESSION MODEL")
lines(pred_poly,col="orange")

plot(test$MEDV,pch=10,col="red",main="RIDGE REGRESSION MODEL")
lines(pred_ridge1,col="purple")


plot(test$MEDV,pch=10,col="red",main="LASSO REGRESSION MODEL")
lines(pred_lasso1,col="dark blue")

plot(test$MEDV,pch=10,col="red",main="ELASTIC NET REGRESSION MODEL")
lines(pred_elastic1,col="dark green")

plot(test$MEDV,pch=10,col="purple",main="POLY RIDGE REGRESSION MODEL")
lines(pred_ridge,col="black")

plot(test$MEDV,pch=10,col="green",main="POLY LASSO REGRESSION MODEL")
lines(pred_lasso,col="blue")

plot(test$MEDV,pch=10,col="blue",main="POLY ELASTIC NET REGRESSION MODEL")
lines(pred_elastic,col="red")

#VISUALIZATION

#RAD
barplot(table(data$RAD),col=5,xlab="Radial Highways",ylab="Count",main="Barplot for Index of Accessibility of Radial Highways") 

#CHAS
barplot(table(data$CHAS),col=4,xlab="Charles River",ylab="Count",main="Barplot CHAS") 

#AGE
hist(data$AGE,breaks=10,col=2,xlab="Age",main="House’s age feature understanding")

#CRIM
hist(data$CRIM,breaks=10,col=3,main="Crime Rate")

#RM
hist(data$RM,breaks=30,col=6,xlab="Rooms",main="Understanding Number of Rooms in the house")




#_______________________________________________________________________________________________________________________________________________

R2_mlrm_train=summary(model_mlrm_train)$r.squared
R2_mlrm_train
R2_step_train=summary(model_mlrm_step)$r.squared
R2_step_train
R_square_test_mlrm
R_square_test_step
R2_poly_train=summary(model_poly)$r.squared
R2_poly_train
R_square_test_poly
R_square_ridge_test
R_square_lasso
R_square_elastic

#_______________________________________________________________________________________________________________________________________________

#VIOLATION OF ASSUMPTIONS

#1. TO CHECK THE NORMALITY OF ERROR
#Error vs Predicted - OUTLIERS and HORIZONTAL BAND FASHION
plot(fitted(model_mlrm),residuals(model_mlrm))
abline(0,0)

#QQ-PLOT of RESIDUALS
qqnorm(residuals(model_mlrm))
qqline(residuals(model_mlrm))

#NORMALITY OF ERRORS/RESIDUALS
plot(density(residuals(model_mlrm)))


head(data)
res=cor(data[,-c(4)],use = "complete.obs")

res2=rcorr(as.matrix(data[,-c(4)]))

corrplot(res, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)
corrplot(res2$r, type="upper", order="hclust", p.mat = res2$P, sig.level = 0.01, insig = "blank")

chart.Correlation(data, histogram=TRUE, pch=19)

#Positive correlations are displayed in blue and negative correlations in red color. Color intensity and the size of the circle are proportional to the correlation coefficients. In the right side of the correlogram, the legend color shows the correlation coefficients and the corresponding colors.

col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = res, col = col, symm = TRUE)


#Distribution of MEDV
plot(density(data$MEDV),main="Distribution Plot for Price of House",col=4,border='red')

#Normal Probability Plot
qqnorm(data$MEDV,main="Probability plot For Price of House",col="blue")
qqline(data$MEDV,col="red")

#Transforming The Revenue to make the distribution Normal
plot(density(log(1+data$MEDV)),main="Distribution Plot for Price of House",col=4,border='red')



#Normal Probability Plot - Transformed
qqnorm(log(1+data$MEDV),main="Probability plot For Revenue",col=3)
qqline(log(1+data$MEDV),col=6)



#_______________________________________________________________________________________________________________________________________________

install.packages("glmnet")
library(glmnet)

x=data.matrix(x_train)
y=data.matrix(y_train)

train_xs=scale(x)

model=glmnet(poly(scale(x),2,raw=TRUE),scale(y),alpha=3.8,fit_intercept=True)
summary(model)

cv_model=cv.glmnet(poly(scale(x),2,raw=TRUE),scale(y),alpha=3.8)
best_lambda=cv_model$lambda.min
best_lambda

best_model=glmnet(poly(train_xs,2,raw=TRUE),scale(y),alpha=3.8,lambda=best_lambda)
coef(best_model)

plot(model,xvar="lambda")

y_predict=predict(model,s=best_lambda,test)
y_predict
SSR=sum((y-y_predict)^2)
SST=sum((y-mean(y))^2)
r2=1-(SSR/SST)
r2
adj_rsq=1-((SSR/SST)*((nrow(y)-1)/(nrow(y)-ncol(data))))
adj_rsq

x=data.matrix(x_test)
y=data.matrix(y_test)

model=glmnet(x,y,alpha=0)
summary(model)

cv_model=cv.glmnet(x,y,alpha=0)
best_lambda=cv_model$lambda.min
best_lambda

best_model=glmnet(x,y,alpha=0,lambda=best_lambda)
coef(best_model)

plot(model,xvar="lambda")

y_predict=predict(model,s=best_lambda,newx=x)
y_predict
SSR=sum((y-y_predict)^2)
SST=sum((y-mean(y))^2)
r2=1-(SSR/SST)
r2
adj_rsq=1-((SSR/SST)*((nrow(y)-1)/(nrow(y)-ncol(data))))
adj_rsq

#_______________________________________________________________________________________________________________________________________________

