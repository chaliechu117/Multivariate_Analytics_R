#라이브러리
install.packages("nnet")
library(nnet)
library(ggplot2)
library(caret)
library(tidyverse)
library(dplyr)

#데이터 전처리 과정
train_data <- read.csv("train_values.csv")
train_label <- read.csv("train_labels.csv")


train_data <- train_data[,c(2:39)]
train_label <- as.factor(train_label[,2])
str(train_data)

categ_idx <- c(8,9,10,11,12,13,14,26)

#question 1 barplot
ggplot(train_data,aes(x=legal_ownership_status  )) +  
  geom_bar()

#onehot encoding
dmy <- dummyVars(~., data = train_data)
train_data <- data.frame(predict(dmy, newdata = train_data))

#dividing data
train_data <- scale(train_data, center = TRUE, scale = TRUE)
data <- data.frame(train_data, Class = train_label)

set.seed(123)
idx_train = sample(1:nrow(data), size = 150000)
data_trn = data[idx_train, ]

data = data[-idx_train,]
idx_valid = sample(1:nrow(data), size = 50000)
data_val=data[idx_valid,]
data_tst=data[-idx_valid,]

#Question 3
perf_eval_multi <- function(cm){
  
  # Simple Accuracy
  ACC = sum(diag(cm))/sum(cm)
  
  # Balanced Correction Rate
  BCR = 1
  for (i in 1:dim(cm)[1]){
    BCR = BCR*(cm[i,i]/sum(cm[i,])) 
  }
  
  BCR = BCR^(1/dim(cm)[1])
  
  return(c(ACC, BCR))
}

# Initialize performance matrix
perf_summary <- matrix(0, nrow = 2, ncol = 2)
colnames(perf_summary) <- c("ACC", "BCR")
rownames(perf_summary) <- c("ANN", "Multilogit")



#ANN------------------------------------------------------------------------
n_var <- dim(data)[2]
ANN_trn_input <- data_trn[,-n_var]
ANN_trn_target <- class.ind(data_trn[,n_var])

ANN_val_input <- data_val[,-n_var]
ANN_val_target <- class.ind(data_val[,n_var])


# Find the best number of hidden nodes in terms of BCR
# Candidate hidden nodes
a <- list(nH = seq(from = 3, to = 30, by = 4),
           maxit = seq(from = 100, to =300 , by = 100)) %>%
  cross_df()

val_perf <- matrix(0,nrow(a),4)


ptm <- proc.time()

for(i in 1:nrow(a)){
  cat("Training ANN: the number of hidden nodes:",a$nH[i],",maxit:",a$maxit[i],"\n")
  evaluation <- c()
  
  # Training the model
  trn_input <- ANN_trn_input
  trn_target <- ANN_trn_target
  tmp_nnet <- nnet(trn_input,trn_target,size = a$nH[i],maxit = a$maxit[i],silent = TRUE,MaxNWts = 10000)
  
  #Evaluate the model
  val_input <- ANN_val_input
  val_target <- ANN_val_target
  
  real <- max.col(val_target)
  pred <- max.col(predict(tmp_nnet,val_input))
  evaluation <- rbind(evaluation,cbind(real,pred))
  #Confusion Matrix
  cfm <- matrix(0,nrow = 3, ncol = 3)
  cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  val_perf[i,1] <- a$nH[i]
  val_perf[i,2] <- a$maxit[i]
  val_perf[i,3:4] <- t(perf_eval_multi(cfm))
}

proc.time() - ptm

#Check best and worst combination of ANN
best_val_perf <- val_perf[order(val_perf[,4],decreasing = TRUE),]
colnames(best_val_perf) <- c("nH","Maxit","ACC","BCR")
best_val_perf

worst_val_perf <- val_perf[order(val_perf[,4],decreasing = FALSE),]
colnames(worst_val_perf) <- c("nH","Maxit","ACC","BCR")
worst_val_perf


#Question 4
best_nH <- best_val_perf[1,1]
best_maxit <- best_val_perf[1,2]

rang = c(0.3,0.5,0.7)
val_perf_rang = matrix(0,length(rang),3)

ptm <- proc.time()

for(i in 1:length(rang)){
  evaluation <- c()
  
  trn_input <- ANN_trn_input
  trn_target <- ANN_trn_target
  tmp_nnet <- nnet(trn_input,trn_target,size = best_nH, maxit = best_maxit, rang = rang[i], MaxNWts = 10000)
  
  val_input <- ANN_val_input
  val_target <- ANN_val_target
  evaluation <- rbind(evaluation, cbind(max.col(val_target),
                                        max.col(predict(tmp_nnet,val_input))))
  
  cfm <- matrix(0,nrow = 3, ncol = 3)
  cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  val_perf_rang[i,1] <- rang[i]
  val_perf_rang[i,2:3] <- t(perf_eval_multi(cfm))
}
cfm
t(perf_eval_multi(cfm))
proc.time() - ptm

best_val_perf_rang <- val_perf_rang[order(val_perf_rang[,3],decreasing = TRUE),]
colnames(best_val_perf_rang) <- c("rang","ACC","BCR")
best_val_perf_rang

best_rang <- best_val_perf_rang[1,1]

##Question 5##
#Combine training and validation dataset
input_idx <- c(1:68)
target_idx <- 69
nnet_trn <- rbind(data_trn,data_val)
nnet_input <- nnet_trn[,input_idx]
nnet_target <- class.ind(nnet_trn[,target_idx])

#Test the ANN
tst_input <- data_tst[,input_idx]
tst_target <- class.ind(data_tst[,target_idx])

val_perf_final <- matrix(0,10,2)
colnames(val_perf_final) <- c("ACC","BCR")


ptm <- proc.time()
for(i in 1:10){
  evluation <- c()
  
  #Train the model
  Final_nnet <- nnet(nnet_trn, nnet_target, size = best_nH, maxit = best_maxit, rang = best_rang, MaxNWts = 10000)
  
  #Test and evaluate the model
  evaluation <- c()
  evaluation <- rbind(evaluation, cbind(max.col(tst_target), max.col(predict(Final_nnet,tst_input))))
  
  Final_cm <- matrix(0,nrow = 3, ncol = 3)
  Final_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  Final_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  Final_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  Final_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  Final_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  Final_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  Final_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  Final_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  Final_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  val_perf_final[,1:2] <- t(perf_eval_multi(Final_cm))
}

proc.time() - ptm

val_perf_final

perf_summary[1,] <- val_perf_final[1,]
perf_summary

#question 6---------------------------------------------------
library(glmnet)
library(GA)
library(tidyverse)
library(corrplot)
library(moments)
library(ggplot2)
library(Epi)
library(dplyr)

perf_eval_multi <- function(cm){
  
  # Simple Accuracy
  ACC = sum(diag(cm))/sum(cm)
  
  # Balanced Correction Rate
  BCR = 1
  for (i in 1:dim(cm)[1]){
    BCR = BCR*(cm[i,i]/sum(cm[i,])) 
  }
  
  BCR = BCR^(1/dim(cm)[1])
  
  return(c(ACC, BCR))
}

# Initialize performance matrix
perf_summary <- matrix(0, nrow = 1, ncol = 2)
colnames(perf_summary) <- c("ACC", "BCR")
rownames(perf_summary) <- c("GA")


fit_F1 <- function(string){
  
  sel_var_idx <- which(string == 1)
  # Use variables whose gene value is 1
  sel_x <- x[, sel_var_idx]
  xy <- data.frame(sel_x, y)
  # Training the model
  GA_nnet <- nnet(sel_x, y, size = 7, maxit = 300, rang = 0.3, MaxNWts = 10000)
  GA_nnet_prey <- predict(GA_nnet, tst_input)
  
  evaluation <- c()
  evaluation <- rbind(evaluation, cbind(max.col(tst_target), max.col(GA_nnet_prey)))
  Final_cm <- matrix(0,nrow = 3, ncol = 3)
  Final_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  Final_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  Final_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  Final_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  Final_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  Final_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  Final_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  Final_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  Final_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  GA_perf <- perf_eval_multi(Final_cm)
  return(GA_perf[1])
}


n_var <- dim(data)[2]
x <- as.matrix(data_trn[,-n_var])
y <- class.ind(data_trn[,n_var])



start_time <- proc.time()
GA_F1 <- ga(type = "binary", fitness = fit_F1, nBits = ncol(x), 
            names = colnames(x), popSize = 50, pcrossover = 0.5, 
            pmutation = 0.01, maxiter = 30, elitism = 1, seed = 1234)
end_time <- proc.time()
end_time - start_time

best_var_idx <- which(GA_F1@solution == 1)
best_var_idx


#Q7---------------------------------------------------------------------
input_idx <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,23,24,27,28,29,30,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47,49,50,52,54,55,56,57,58,59,60,61,62,65,67,68)
target_idx <- 69

nnet_trn <- rbind(data_trn,data_val)
nnet_input <- nnet_trn[,input_idx]
nnet_target <- class.ind(nnet_trn[,target_idx])

#Test the ANN
tst_input <- data_tst[,input_idx]
tst_target <- class.ind(data_tst[,target_idx])

val_perf_final <- matrix(0,10,2)
colnames(val_perf_final) <- c("ACC","BCR")




ptm <- proc.time()
for(i in 1:10){
  evluation <- c()
  
  #Train the model
  Final_nnet <- nnet(nnet_trn, nnet_target, size = 7, maxit = 300, rang = 0.3, MaxNWts = 10000)
  
  #Test and evaluate the model
  evaluation <- c()
  evaluation <- rbind(evaluation, cbind(max.col(tst_target), max.col(predict(Final_nnet,tst_input))))
  
  Final_cm <- matrix(0,nrow = 3, ncol = 3)
  Final_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  Final_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  Final_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  Final_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  Final_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  Final_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  Final_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  Final_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  Final_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  val_perf_final[,1:2] <- t(perf_eval_multi(Final_cm))
}

proc.time() - ptm

val_perf_final


# Initialize performance matrix
perf_summary <- matrix(0, nrow = 1, ncol = 2)
colnames(perf_summary) <- c("ACC", "BCR")
rownames(perf_summary) <- c("New ANN")

perf_summary[1,] <- val_perf_final[1,]
perf_summary

#8--------------------------------------------------------
input_idx <- c(1:68)
target_idx <- 69
nnet_trn <- rbind(data_trn,data_val)
nnet_input <- nnet_trn[,input_idx]
nnet_target <- as.factor(nnet_trn[,target_idx])

perf_eval_multi <- function(cm){
  
  # Simple Accuracy
  ACC = sum(diag(cm))/sum(cm)
  
  # Balanced Correction Rate
  BCR = 1
  for (i in 1:dim(cm)[1]){
    BCR = BCR*(cm[i,i]/sum(cm[i,])) 
  }
  
  BCR = BCR^(1/dim(cm)[1])
  
  return(c(ACC, BCR))
}

# Initialize performance matrix
perf_summary <- matrix(0, nrow = 3, ncol = 2)
colnames(perf_summary) <- c("ACC", "BCR")
rownames(perf_summary) <- c("non", "post", "pre")


library(tree)
CART_post <- tree(Class ~ ., nnet_trn)
summary(CART_post)

plot(CART_post)
text(CART_post, pretty = 1)

CART_post_pray <- predict(CART_post, data_tst, type = "class")
CART_post_cm <- table(data_tst$Class, CART_post_pray)
CART_post_cm
perf_summary[1,(1:2)] <- perf_eval_multi(CART_post_cm)

#Post pruning
CART_post_cv <- cv.tree(CART_post, FUN = prune.misclass)
plot(CART_post_cv$size, CART_post_cv$dev, type = "b")
CART_post_cv
CART_post_pruned <- prune.misclass(CART_post, best = 5)
CART_post_prey <- predict(CART_post_pruned, data_tst, type = "class")
CART_post_cm <- table(data_tst$Class, CART_post_prey)
CART_post_cm

perf_summary[2,] <- perf_eval_multi(CART_post_cm)

#Pre-pruning
library(party)
library(ROCR)
min_criterion = c(0.9, 0.95, 0.99)
min_split = c(10, 30, 50, 100)
max_depth = c(0, 10, 5)

str(data_trn)

CART_pre_search_result = matrix(0,length(min_criterion)*length(min_split)*length(max_depth),11)
colnames(CART_pre_search_result) <- c("min_criterion", "min_split", "max_depth", 
                                      "TPR", "Precision", "TNR", "ACC", "BCR", "F1", "AUROC", "N_leaves")
#memory.size(max = TRUE)    
#memory.size(max = FALSE)  
#memory.limit(size = NA) 
#memory.limit(size = 50000) 
perf_eval <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}

# Performance table
Perf_Table <- matrix(0, nrow = 1, ncol = 6)
rownames(Perf_Table) <- c( "Pre-Pruning")
colnames(Perf_Table) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-Measure")
Perf_Table



input_idx <- c(1:68)
target_idx <- 69

nnet_input <- data_trn[,input_idx]
nnet_target <- (data_trn[,target_idx])
nnet_trn <- data.frame(nnet_input, nnet_target)


nnet_input <- data_val[,input_idx]
nnet_target <- (data_val[,target_idx])
nnet_val <- data.frame(nnet_input, nnet_target)
gc()
iter_cnt = 1

for (i in 1:length(min_criterion)){
  for ( j in 1:length(min_split)){
    for ( k in 1:length(max_depth)){
      
      cat("CART Min criterion:", min_criterion[i], ", Min split:", min_split[j], ", Max depth:", max_depth[k], "\n")
      tmp_control = ctree_control(mincriterion = min_criterion[i], minsplit = min_split[j], maxdepth = max_depth[k])
      tmp_tree <- ctree(Class ~ ., data = data_trn, controls = tmp_control)
      tmp_tree_val_prediction <- predict(tmp_tree, newdata = data_val)
      tmp_tree_val_response <- treeresponse(tmp_tree, newdata = data_val)
      tmp_tree_val_prob <- 1-unlist(tmp_tree_val_response, use.names=F)[seq(1,nrow(data_val)*2,2)]
      #tmp_tree_val_rocr <- prediction(tmp_tree_val_prob, data_val$Class)
      # Confusion matrix for the validation dataset
      tmp_tree_val_cm <- table(data_val$Class, tmp_tree_val_prediction)
      
      # parameters
      CART_pre_search_result[iter_cnt,1] = min_criterion[i]
      CART_pre_search_result[iter_cnt,2] = min_split[j]
      CART_pre_search_result[iter_cnt,3] = max_depth[k]
      # Performances from the confusion matrix
      CART_pre_search_result[iter_cnt,4:9] = perf_eval(tmp_tree_val_cm)
      # AUROC
      #CART_pre_search_result[iter_cnt,10] = unlist(performance(tmp_tree_val_rocr, "auc")@y.values)
      # Number of leaf nodes
      #CART_pre_search_result[iter_cnt,11] = length(nodes(tmp_tree, unique(where(tmp_tree))))
      iter_cnt = iter_cnt + 1
    }
  }
}

CART_pre_search_result <- CART_pre_search_result[order(CART_pre_search_result[,7], decreasing = T),]
CART_pre_search_result
best_criterion <- CART_pre_search_result[1,1]
best_split <- CART_pre_search_result[1,2]
best_depth <- CART_pre_search_result[1,3]

tree_control = ctree_control(mincriterion = best_criterion, minsplit = best_split, maxdepth = best_depth)

CART_trn <- rbind(data_trn, data_val)
CART_pre <- ctree(Class ~ ., data = CART_trn, controls = tree_control)
CART_pre_prediction <- predict(CART_pre, newdata = data_tst)
CART_pre_response <- treeresponse(CART_pre, newdata = data_tst)
CART_pre_cm <- table(data_tst$Class, CART_pre_prediction)
CART_pre_cm

CART_pre_cm <- table(data_tst$Class, CART_pre_prediction)
CART_pre_cm

Perf_Table <- matrix(0, nrow = 1, ncol = 6)
rownames(Perf_Table) <- c( "Pre-Pruning")
colnames(Perf_Table) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-Measure")
Perf_Table

perf_summary <- matrix(0, nrow = 1, ncol = 2)
colnames(perf_summary) <- c("ACC", "BCR")
rownames(perf_summary) <- c("Decision Tree")

Perf_Table[1,] <- perf_eval(CART_pre_cm)
Perf_Table

perf_summary
#Question 9---------------------------------------------------------------------------------------------------
#Train multinomial logistic regression
input_idx <- c(1:68)
target_idx <- 69
nnet_trn <- rbind(data_trn,data_val)
nnet_input <- nnet_trn[,input_idx]
nnet_target <- class.ind(nnet_trn[,target_idx])

ml_logit <- multinom(Class ~ ., data = nnet_trn)

summary(ml_logit)
t(summary(ml_logit)$coefficients)

ml_logit_pred <- predict(ml_logit, newdata = data_tst)
cfmatrix <- table(data_tst$Class,ml_logit_pred)
cfmatrix

perf_summary <- matrix(0, nrow = 1, ncol = 2)
colnames(perf_summary) <- c("ACC", "BCR")
rownames(perf_summary) <- c("Multi logit")

perf_summary[1,] <- perf_eval_multi(cfmatrix)
perf_summary











