install.packages("glmnet")
install.packages("GA")
library(glmnet)
library(GA)

perf_eval <- function(tgt_y, pre_y){
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


Perf_Table <- matrix(0, nrow = 8, ncol = 3)
rownames(Perf_Table) <- c("All", "Forward", "Backward", "Stepwise", "GA", "Ridge", "Lasso", "Elastic Net")
colnames(Perf_Table) <- c("RMSE","MAE","MAPE")

# Load the data & Preprocessing
Ploan <- read.csv("Concrete_Data.csv")

Ploan_input <- Ploan[,-c(9)]
Ploan_input_scaled <- scale(Ploan_input, center = TRUE, scale = TRUE)
Ploan_target <- Ploan$csMPa

Ploan_data_scaled <- data.frame(Ploan_input_scaled, Ploan_target)

# Split the data into the training/validation sets
set.seed(12345) 
trn_idx <- sample(1:nrow(Ploan_data_scaled), round(0.7*nrow(Ploan_data_scaled)))
Ploan_trn <- Ploan_data_scaled[trn_idx,]
Ploan_tst <- Ploan_data_scaled[-trn_idx,]

# Variable selection method 0
full_model <- lm(Ploan_target ~ ., data = Ploan_trn)
summary(full_model)
full_model_coeff <- as.matrix(full_model$coefficients, 9, 1)
full_model_coeff

full_model_prey <- predict(full_model, newdata = Ploan_tst)
Perf_Table[1,] <- perf_eval(Ploan_tst$Ploan_target, full_model_prey)

