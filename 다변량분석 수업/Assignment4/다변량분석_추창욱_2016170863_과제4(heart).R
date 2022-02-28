library(tidyverse)
library(corrplot)
library(moments)
library(ggplot2)
library(Epi)
# Performance Evaluation Function -----------------------------------------
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
Perf_Table <- matrix(0, nrow = 3, ncol = 6)
rownames(Perf_Table) <- c("Non-Pruning","Post-Pruning", "Pre-Pruning")
colnames(Perf_Table) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-Measure")
Perf_Table

# Load the data & Preprocessing
voice <- read.csv("heart.csv")
names(voice)[1]<-c("Age")
input_idx <- c(1:13)
target_idx <- 14


# Conduct the normalization
voice_input <- voice[,input_idx]
voice_input <- scale(voice_input, center = TRUE, scale = TRUE)
voice_target <- as.factor(voice[,target_idx])
voice_data <- data.frame(voice_input, voice_target)

# Split the data into the training/validation sets
set.seed(12345)
trn_idx <- sample(1:nrow(voice_data), round(0.6*nrow(voice_data)))
voice_trn <- voice_data[trn_idx,]
voice_tst <- voice_data[-trn_idx,]

# CART with Non-Pruning -------------------------------
library(tree)
# Training the tree
CART_post <- tree(voice_target ~ ., voice_trn)
summary(CART_post)

#plot the tree
plot(CART_post)
text(CART_post, pretty = 1)

# Prediction
CART_post_pray <- predict(CART_post, voice_tst, type = "class")
CART_post_cm <- table(voice_tst$voice_target, CART_post_pray)
CART_post_cm
Perf_Table[1,(1:6)] <- perf_eval(CART_post_cm)
Perf_Table

# Plot the ROC
t <-CART_post_pray
head(t)
ROC(test=t, stat=voice_tst$voice_target, plot='ROC', AUC=T, main="Non-Pruning ROC")


# CART with Post-Pruning -------------------------------

library(tree)

# Training the tree
CART_post <- tree(voice_target ~ ., voice_trn)
summary(CART_post)

# Plot the tree
plot(CART_post)
text(CART_post, pretty = 1)

# Find the best tree
set.seed(12345)
CART_post_cv <- cv.tree(CART_post, FUN = prune.misclass)

# Plot the pruning result
plot(CART_post_cv$size, CART_post_cv$dev, type = "b")
CART_post_cv

# Select the final model
CART_post_pruned <- prune.misclass(CART_post, best = 10)
plot(CART_post_pruned)
text(CART_post_pruned, pretty = 1)

# Prediction
CART_post_prey <- predict(CART_post_pruned, voice_tst, type = "class")
CART_post_cm <- table(voice_tst$voice_target, CART_post_prey)
CART_post_cm

Perf_Table[2,] <- perf_eval(CART_post_cm)
Perf_Table

# Plot the ROC
t <-CART_post_prey
head(t)
ROC(test=t, stat=voice_tst$voice_target, plot='ROC', AUC=T, main="Post-Pruning ROC")

# CART with Pre-Pruning -------------------------------
# For CART

library(party)

# For AUROC

library(ROCR)

# Divide the dataset into training/validation/test datasets
voice <- read.csv("heart.csv")
names(voice)[1]<-c("Age")
input_idx <- c(1:13)
target_idx <- 14

voice_input <- voice[,input_idx]
voice_target <- as.factor(voice[,target_idx])
voice_data <- data.frame(voice_input, voice_target)

#split data
set.seed(12345)
trn_idx <- sample(1:nrow(voice_data), round(0.5*nrow(voice_data)))
voice_trn <- voice_data[trn_idx,]
voice_x <- voice_data[-trn_idx,]

set.seed(12345)
val_idx <- sample(1:nrow(voice_x), round(0.2*nrow(voice_x)))
voice_val <- voice_x[val_idx,]
voice_tst <- voice_x[-val_idx,]

# Construct single tree and evaluation
# tree parameter settings
min_criterion = c(0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
min_split = c(10,20,30,40,50,100)
max_depth = c(0, 5, 10, 15, 20)

CART_pre_search_result = matrix(0,length(min_criterion)*length(min_split)*length(max_depth),11)
colnames(CART_pre_search_result) <- c("min_criterion", "min_split", "max_depth", 
                                      "TPR", "Precision", "TNR", "ACC", "BCR", "F1", "AUROC", "N_leaves")

iter_cnt = 1

for (i in 1:length(min_criterion)){
  for ( j in 1:length(min_split)){
    for ( k in 1:length(max_depth)){
      
      cat("CART Min criterion:", min_criterion[i], ", Min split:", min_split[j], ", Max depth:", max_depth[k], "\n")
      tmp_control = ctree_control(mincriterion = min_criterion[i], minsplit = min_split[j], maxdepth = max_depth[k])
      tmp_tree <- ctree(voice_target ~ ., data = voice_trn, controls = tmp_control)
      tmp_tree_val_prediction <- predict(tmp_tree, newdata = voice_val)
      tmp_tree_val_response <- treeresponse(tmp_tree, newdata = voice_val)
      tmp_tree_val_prob <- 1-unlist(tmp_tree_val_response, use.names=F)[seq(1,nrow(voice_val)*2,2)]
      tmp_tree_val_rocr <- prediction(tmp_tree_val_prob, voice_val$voice_target)
      # Confusion matrix for the validation dataset
      tmp_tree_val_cm <- table(voice_val$voice_target, tmp_tree_val_prediction)
      
      # parameters
      CART_pre_search_result[iter_cnt,1] = min_criterion[i]
      CART_pre_search_result[iter_cnt,2] = min_split[j]
      CART_pre_search_result[iter_cnt,3] = max_depth[k]
      # Performances from the confusion matrix
      CART_pre_search_result[iter_cnt,4:9] = perf_eval(tmp_tree_val_cm)
      # AUROC
      CART_pre_search_result[iter_cnt,10] = unlist(performance(tmp_tree_val_rocr, "auc")@y.values)
      # Number of leaf nodes
      CART_pre_search_result[iter_cnt,11] = length(nodes(tmp_tree, unique(where(tmp_tree))))
      iter_cnt = iter_cnt + 1
    }
  }
}

# Find the best set of parameters
CART_pre_search_result <- CART_pre_search_result[order(CART_pre_search_result[,10], decreasing = T),]
CART_pre_search_result
best_criterion <- CART_pre_search_result[1,1]
best_split <- CART_pre_search_result[1,2]
best_depth <- CART_pre_search_result[1,3]

# Construct the best tree
tree_control = ctree_control(mincriterion = best_criterion, minsplit = best_split, maxdepth = best_depth)

# Use the training and validation dataset to train the best tree
CART_trn <- rbind(voice_trn, voice_val)

CART_pre <- ctree(voice_target ~ ., data = voice_trn, controls = tree_control)
CART_pre_prediction <- predict(CART_pre, newdata = voice_tst)
CART_pre_response <- treeresponse(CART_pre, newdata = voice_tst)

# Plot the best tree
plot(CART_pre)
plot(CART_pre, type="simple")

# Performance of the best tree
CART_pre_cm <- table(voice_tst$voice_target, CART_pre_prediction)
CART_pre_cm

Perf_Table[3,] <- perf_eval(CART_pre_cm)
Perf_Table

# Plot the ROC
t <-CART_pre_prediction
head(t)
ROC(test=t, stat=voice_tst$voice_target, plot='ROC', AUC=T, main="Pre-Pruning ROC")

# Plot the best tree
plot(CART_pre)
plot(CART_pre, type="simple")

# logistic Regression -------------------------------
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
rownames(Perf_Table) <- c( "Logistic Regression")
colnames(Perf_Table) <- c("TPR", "Precision", "TNR", "Accuracy", "BCR", "F1-Measure")
Perf_Table

heart <- read.csv("heart.csv")
names(heart)[1]<-c("Age")
input_idx <- c(1:13)
target_idx <- 14


# Conduct the normalization
heart_input <- heart[,input_idx]
heart_input <- scale(heart_input, center = TRUE, scale = TRUE)
heart_target <- as.factor(heart[,target_idx])
heart_data <- data.frame(heart_input, heart_target)

# Split the data into the training/validation sets
set.seed(12345)
trn_idx <- sample(1:nrow(heart_data), round(0.6*nrow(heart_data)))
heart_trn <- heart_data[trn_idx,]
heart_tst <- heart_data[-trn_idx,]

# Train the Logistic Regression Model with all variables
full_lr <- glm(heart_target ~ ., family=binomial, heart_trn)
summary(full_lr)

lr_response <- predict(full_lr, type = "response", newdata = heart_tst)
lr_target <- heart_tst$heart_target
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >= 0.5)] <- 1
cm_full <- table(lr_target, lr_predicted)
cm_full

Perf_Table[1,] <- perf_eval(cm_full)
Perf_Table

# Plot the ROC
t <-lr_response
head(t)
ROC(test=t, stat=heart_tst$heart_target, plot='ROC', AUC=T, main="Logistic Regression ROC")