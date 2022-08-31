install.packages("moments")
install.packages("tidyverse")
library(moments)
library(tidyverse)
library(corrplot)

perf_eval_reg <- function(tgt_y, pre_y){
  # RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  # MAE
  mae <- mean(abs(tgt_y - pre_y))
  # MAPE
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  return(c(rmse, mae, mape))
}
# Initialize a performance summary table
perf_mat <- matrix(0, nrow = 2, ncol = 3)
rownames(perf_mat) <-c("바꾸기 전","바꾼 후")
colnames(perf_mat)<-c("RMSE","MAE","MAPE")
perf_mat

# Dataset 1: cement
insurance<- read.csv("Concrete_Data_Yeh.csv")

#데이터 분석하기
boxplot(insurance$fineaggregate)
qqnorm(insurance$fineaggregate)
qqline(insurance$fineaggregate)
mean(insurance$fineaggregate)
sd(insurance$fineaggregate)
skewness(insurance$fineaggregate)
kurtosis(insurance$fineaggregate)

#이상치 데이터 삭제하기
summary(insurance$slag)
a<-which(insurance$slag>summary(insurance$slag)[5] + 1.5*IQR(insurance$slag))

summary(insurance$water)
b<-which(insurance$water>summary(insurance$water)[5] + 1.5*IQR(insurance$water))
c<-which(insurance$water<summary(insurance$water)[2] - 1.5*IQR(insurance$water))

summary(insurance$superplasticizer)
d<-which(insurance$superplasticizer>summary(insurance$superplasticizer)[5] + 1.5*IQR(insurance$superplasticizer))

summary(insurance$fineaggregate)
e<-which(insurance$fineaggregate>summary(insurance$fineaggregate)[5] + 1.5*IQR(insurance$fineaggregate))

#이상치 제거
insurance <- insurance[-c(a,b,c,d,e),]

#확인
summary(insurance$slag)
which(insurance$slag>summary(insurance$slag)[5] + 1.5*IQR(insurance$slag))
summary(insurance$water)
which(insurance$water>summary(insurance$water)[5] + 1.5*IQR(insurance$water))
which(insurance$water<summary(insurance$water)[2] - 1.5*IQR(insurance$water))

#상관관계
plot(insurance[,1:8])
cor(insurance[,1:8])
x<-cor(insurance[,1:8])
corrplot(x)

# Indices for the activated input variables
nCar <- nrow(insurance)
nVar <- ncol(insurance)



#prepare data
insurance_mlr_data <- cbind(insurance)


# Split the data into the training/validation sets
set.seed(12345) 
insurance_trn_idx <- sample(1:nCar, round(0.7*nCar))
insurance_trn_data <- insurance_mlr_data[insurance_trn_idx,]
insurance_val_data <- insurance_mlr_data[-insurance_trn_idx,]

# Train the MLR
mlr_insurance <- lm(csMPa ~ ., data = insurance_trn_data)
mlr_insurance
summary(mlr_insurance)
plot(mlr_insurance)


# Plot the result
plot(insurance_trn_data$csMPa, fitted(mlr_insurance), 
     xlim = c(0,100), ylim = c(0,100))
abline(0,1,lty=3)

# normality test of residuals
insurance_resid <- resid(mlr_insurance)

m <- mean(insurance_resid)
std <- sqrt(var(insurance_resid))

hist(insurance_resid, density=20, breaks=50, prob=TRUE, 
     xlab="x-variable", main="normal curve over histogram")

curve(dnorm(x, mean=m, sd=std), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

skewness(insurance_resid)
kurtosis(insurance_resid)

#종속변수와 0.01유의미한 변수들 간의 상관관계
insurance_remake<- insurance[ ,-c(5,6,7)]
cor(insurance_remake[,1:6])
insurance_remake2<-insurance[,c(1,2,4,8)]

# Performance Measure
mlr_insurance_haty <- predict(mlr_insurance, newdata = insurance_val_data)

perf_mat[1,] <- perf_eval_reg(insurance_val_data$csMPa, mlr_insurance_haty)
perf_mat



#New model
insurance_remake2<-insurance[,c(1,2,4,8,9)]
# Indices for the activated input variables
nCar <- nrow(insurance_remake2)
nVar <- ncol(insurance_remake2)



#prepare data
insurance_remake2_mlr_data <- cbind(insurance_remake2)


# Split the data into the training/validation sets
set.seed(12345) 
insurance_remake2_trn_idx <- sample(1:nCar, round(0.7*nCar))
insurance_remake2_trn_data <- insurance_remake2_mlr_data[insurance_remake2_trn_idx,]
insurance_remake2_val_data <- insurance_remake2_mlr_data[-insurance_remake2_trn_idx,]

# Train the MLR
mlr_insurance_remake2 <- lm(csMPa ~ ., data = insurance_remake2_trn_data)
mlr_insurance_remake2
summary(mlr_insurance_remake2)
plot(mlr_insurance_remake2)

mlr_insurance_remake2_haty <- predict(mlr_insurance_remake2, newdata = insurance_remake2_val_data)

perf_mat[2,] <- perf_eval_reg(insurance_remake2_val_data$csMPa, mlr_insurance_remake2_haty)
perf_mat

#GAM analysis
require(gam)
gam1<-gam(csMPa~s(cement,df=4)+s(slag,df=4)+s(water,df=4)+s(age,df=4) ,data = insurance_remake2)
summary(gam1)

par(mfrow=c(1,4))
plot(gam1,se = TRUE)
