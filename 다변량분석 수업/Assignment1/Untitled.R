install.packages("moments")
library(moments)

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

# Dataset 1: bike
bike <- read.csv("hour.csv")

# Indices for the activated input variables
nCar <- nrow(bike)
nVar <- ncol(bike)
id_idx <- c(1,2)

# individual plotting
boxplot(bike$windspeed)
mean(bike$windspeed)
sd(bike$windspeed)
skewness(bike$windspeed)
kurtosis(bike$windspeed)

#correlation
install.packages("corrplot")
library(corrplot)
#scatter plot matrix
bike_mlr_data.cor <- cor(bike_mlr_data)
bike_mlr_data.cor
corrplot(bike_mlr_data.cor)

# dummy
bike_mlr_data <- cbind(bike[,-c(id_idx,7,4,12,14,8,15,16)])


#dummify
#weathersit
bike_mlr_data$w1 <-ifelse(bike_mlr_data$weathersit==1,1,0)
bike_mlr_data$w2 <-ifelse(bike_mlr_data$weathersit==2,1,0)
bike_mlr_data$w3 <-ifelse(bike_mlr_data$weathersit==3,1,0)

bike_mlr_data <- cbind(bike_mlr_data[,-c(5)])

#season
bike_mlr_data$s1 <-ifelse(bike_mlr_data$season==1,1,0)
bike_mlr_data$s2 <-ifelse(bike_mlr_data$season==2,1,0)
bike_mlr_data$s3 <-ifelse(bike_mlr_data$season==3,1,0)
bike_mlr_data <- cbind(bike_mlr_data[,-c(1)])

#mnth
bike_mlr_data$m1 <-ifelse(bike_mlr_data$mnth==1,1,0)
bike_mlr_data$m2 <-ifelse(bike_mlr_data$mnth==2,1,0)
bike_mlr_data$m3 <-ifelse(bike_mlr_data$mnth==3,1,0)
bike_mlr_data$m4 <-ifelse(bike_mlr_data$mnth==4,1,0)
bike_mlr_data$m5 <-ifelse(bike_mlr_data$mnth==5,1,0)
bike_mlr_data$m6 <-ifelse(bike_mlr_data$mnth==6,1,0)
bike_mlr_data$m7 <-ifelse(bike_mlr_data$mnth==7,1,0)
bike_mlr_data$m8 <-ifelse(bike_mlr_data$mnth==8,1,0)
bike_mlr_data$m9 <-ifelse(bike_mlr_data$mnth==9,1,0)
bike_mlr_data$m10 <-ifelse(bike_mlr_data$mnth==10,1,0)
bike_mlr_data$m11 <-ifelse(bike_mlr_data$mnth==11,1,0)

bike_mlr_data <- cbind(bike_mlr_data[,-c(1)])


#hour
bike_mlr_data$h1 <-ifelse(bike_mlr_data$hr==1,1,0)
bike_mlr_data$h2 <-ifelse(bike_mlr_data$hr==2,1,0)
bike_mlr_data$h3 <-ifelse(bike_mlr_data$hr==3,1,0)
bike_mlr_data$h4 <-ifelse(bike_mlr_data$hr==4,1,0)
bike_mlr_data$h5 <-ifelse(bike_mlr_data$hr==5,1,0)
bike_mlr_data$h6 <-ifelse(bike_mlr_data$hr==6,1,0)
bike_mlr_data$h7 <-ifelse(bike_mlr_data$hr==7,1,0)
bike_mlr_data$h8 <-ifelse(bike_mlr_data$hr==8,1,0)
bike_mlr_data$h9 <-ifelse(bike_mlr_data$hr==9,1,0)
bike_mlr_data$h10 <-ifelse(bike_mlr_data$hr==10,1,0)
bike_mlr_data$h11 <-ifelse(bike_mlr_data$hr==11,1,0)
bike_mlr_data$h12 <-ifelse(bike_mlr_data$hr==12,1,0)
bike_mlr_data$h13 <-ifelse(bike_mlr_data$hr==13,1,0)
bike_mlr_data$h14 <-ifelse(bike_mlr_data$hr==14,1,0)
bike_mlr_data$h15 <-ifelse(bike_mlr_data$hr==15,1,0)
bike_mlr_data$h16 <-ifelse(bike_mlr_data$hr==16,1,0)
bike_mlr_data$h17 <-ifelse(bike_mlr_data$hr==17,1,0)
bike_mlr_data$h18 <-ifelse(bike_mlr_data$hr==18,1,0)
bike_mlr_data$h19 <-ifelse(bike_mlr_data$hr==19,1,0)
bike_mlr_data$h20 <-ifelse(bike_mlr_data$hr==20,1,0)
bike_mlr_data$h21 <-ifelse(bike_mlr_data$hr==21,1,0)
bike_mlr_data$h22 <-ifelse(bike_mlr_data$hr==22,1,0)
bike_mlr_data$h23 <-ifelse(bike_mlr_data$hr==23,1,0)

bike_mlr_data <- cbind(bike_mlr_data[,-c(1)])
bike_mlr_data <- cbind(bike_mlr_data[,-c(1,5,6,7,11,12,13,14,15,16,17,18,20,21,22)])
# test <-> train compare
set.seed(12345)
bike_trn_idx <- sample(1:nCar, round(0.7*nCar))
bike_trn_data <- bike_mlr_data[bike_trn_idx,]
bike_val_data <- bike_mlr_data[-bike_trn_idx,]

#conclusion push
mlr_bike <- lm(cnt ~ ., data = bike_trn_data)
mlr_bike
summary(mlr_bike)
plot(mlr_bike)


# Plot the result
plot(bike_trn_data$quality, fitted(mlr_bike), xlim = c(0,10),
     ylim = c(0,10))
abline(0,1,lty=3)

# normality test of residuals
bike_resid <- resid(mlr_bike)
m <- mean(bike_resid)
std <- sqrt(var(bike_resid))
hist(bike_resid, density=20, breaks=50, prob=TRUE, xlab="x-variable",
     main="normal curve over histogram")
curve(dnorm(x, mean=m, sd=std), col="darkblue", lwd=2, add=TRUE, yaxt="n")
skewness(bike_resid)
kurtosis(bike_resid)

# Performance Measure
mlr_bike_haty <- predict(mlr_bike, newdata = bike_trn_data)
perf_mat[1,] <- perf_eval_reg(bike_trn_data$cnt, mlr_bike_haty)
perf_mat

library()
vif <- vif(lm(bike_trn_idx, data=cnt))
