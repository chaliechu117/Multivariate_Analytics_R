Damage <- read.csv("Earthquake_Damage.csv")
#Reduce Data set
library(dplyr)
#for real sampling
class_1 <- sample_n(Damage[which(Damage$damage_grade == 1),],2520)
class_2 <- sample_n(Damage[which(Damage$damage_grade == 2),],14825)
class_3 <- sample_n(Damage[which(Damage$damage_grade == 3),],8725)

#for under sampling
#class_1 <- sample_n(Damage[which(Damage$damage_grade == 1),],2520)
#class_2 <- sample_n(Damage[which(Damage$damage_grade == 2),],2520)
#class_3 <- sample_n(Damage[which(Damage$damage_grade == 3),],2520)

Damage <- rbind(class_1,class_2,class_3)

##Data Preparation##
library(nnet)
cha_idx <- c(9:15,27)
character <- Damage[,cha_idx]
a <- names(character)

frame <- 0
for(i in 1:length(character)){
  tmp <- class.ind(character[,i])
  for(j in 1:ncol(tmp)){
    colnames(tmp)[j] <-  paste0(a[i],"_",colnames(tmp)[j])
  }
  assign(paste0("dummy_",a[i]),tmp)
  frame <- cbind(frame,tmp)
}

cha_input <- frame[,c(2:39)]

num_idx <- c(2:8,28)
num_input <- scale(Damage[,num_idx], center = TRUE, scale = TRUE)

bin_idx <- c(16:26,29:39)
bin_input <- lapply(Damage[,bin_idx],factor)

input <- data.frame(cha_input,num_input,bin_input)
target <- as.factor(Damage[,c(40)])
Final_data <- data.frame(input, Class = target)

#Split the Data
#set.seed(12345)
trn <- Final_data[sample(nrow(Final_data),7430),]
val <- Final_data[sample(nrow(Final_data),2477),]
tst <- Final_data[sample(nrow(Final_data),3128),]

#for undersampling
#set.seed(12345)
#trn <- Final_data[sample(nrow(Final_data),4310),]
#val <- Final_data[sample(nrow(Final_data),1436),]
#tst <- Final_data[sample(nrow(Final_data),1814),]

#Performance Evaluation
perf_eval_multi <- function(cm){
  #Simple Accuracy
  ACC <- sum(diag(cm))/sum(cm)
  #Balanced Correction Rate
  BCR <- 1
  for(i in 1:dim(cm)[1]){
    BCR = BCR * (cm[i,i]/sum(cm[i,]))
  }
  BCR = BCR^(1/dim(cm)[1])
  return(c(ACC,BCR))
}

perf_table <- matrix(0,nrow = 8, ncol = 2)
colnames(perf_table) <- c("ACC","BCR")
rownames(perf_table) <- c("MLR","CART","ANN","Bagging CART","Random Forests","Bagging ANN","AdaBoost","GBM")

#Question 1
# Multinomial Logistic Regression----------------------------------
trn_data <- rbind(trn,val)

ptm <- proc.time()
ml_logit <- multinom(Class ~ ., data = trn_data)
MLR.Time <- proc.time() - ptm
MLR.Time

summary(ml_logit)
t(summary(ml_logit)$coefficients)

ml_logit_prey <- predict(ml_logit, newdata = tst)
mlr_cfm <- table(tst$Class, ml_logit_prey)
mlr_cfm

perf_table[1,] <- perf_eval_multi(mlr_cfm)
perf_table

#Classification and Regression Tree
library(party)
library(tidyverse)
library(dplyr)

input_idx <- c(1:68)
target_idx <- 69

set.seed(12345)
CART_trn <- data.frame(trn[,input_idx], GradeYN = trn[,target_idx])
CART_val <- data.frame(val[,input_idx], GradeYN = val[,target_idx])
CART_tst <- data.frame(tst[,input_idx], GradeYN = tst[,target_idx])

gs_CART <- list(min_criterion = c(0.8,0.9,0.99),min_split = c(10,50,100),max_depth = c(0,5,10,20)) %>%
  cross_df()

CART_result = matrix(0,nrow(gs_CART),5)
colnames(CART_result) <- c("min_criteron","min_split","max_depth","ACC","BCR")

iter_cnt = 1
for(i in 1:nrow(gs_CART)){
  cat("CART Min Criterion:",gs_CART$min_criterion[i],",Minsplit:",gs_CART$min_split[i],",Max depth:",gs_CART$max_depth[i],"\n")
  
  tmp_control = ctree_control(mincriterion = gs_CART$min_criterion[i],minsplit = gs_CART$min_split[i],maxdepth = gs_CART$max_depth[i])
  tmp_tree <- ctree(GradeYN ~., data = CART_trn, controls = tmp_control)
  tmp_tree_val_prediction <- predict(tmp_tree, newdata = CART_val)
  
  tmp_tree_val_cm <- table(CART_val$GradeYN, tmp_tree_val_prediction)
  tmp_tree_val_cm
  
  CART_result[iter_cnt,1] = gs_CART$min_criterion[i]
  CART_result[iter_cnt,2] = gs_CART$min_split[i]
  CART_result[iter_cnt,3] = gs_CART$max_depth[i]
  
  CART_result[iter_cnt,4:5] = perf_eval_multi(tmp_tree_val_cm)
  iter_cnt = iter_cnt + 1
}

CART_result <- CART_result[order(CART_result[,5],decreasing = T),]
CART_result

best_criterion <- CART_result[1,1]
best_split <- CART_result[1,2]
best_depth <- CART_result[1,3]

tree_control = ctree_control(mincriterion = best_criterion, minsplit = best_split, maxdepth = best_depth)
CART_data <- rbind(CART_trn,CART_val)

ptm <- proc.time()
CART_final <- ctree(GradeYN ~., data = CART_data, controls = tree_control)
CART.Time <- proc.time() - ptm
CART.Time

CART_prediction <- predict(CART_final, newdata = CART_tst)
CART_cm <- table(CART_tst$GradeYN, CART_prediction)
CART_cm

perf_table[2,] <- perf_eval_multi(CART_cm)
perf_table

plot(CART_final)

#Artificial Neural Network
library(tidyverse)
gs_ANN <- list(nH = seq(from = 10, to = 40, by = 10),
               maxit = seq(from = 100, to =300 , by = 50),
               rang = c(0.3, 0.5, 0.7, 0.9)) %>%
  cross_df()

input <- trn[,input_idx] 
target <- class.ind(trn[,target_idx]) 
val_input <- val[,input_idx]
val_target <- class.ind(val[,target_idx]) 

data <- rbind(trn,val) 
data_input <- data[,input_idx]
data_target <- class.ind(data[,target_idx])
tst_input <- tst[,input_idx]
tst_target <- class.ind(tst[,target_idx])

ANN_result <- matrix(0,nrow(gs_ANN),5)

set.seed(12345)
for(i in 1:nrow(gs_ANN)){
  cat("Training ANN: the number of hidden nodes:",gs_ANN$nH[i],",maxit:",gs_ANN$maxit[i],",rang:",gs_ANN$rang[i],"\n")
  evaluation <- c()
  
  # Training the model
  tmp_nnet <- nnet(input,target,size = gs_ANN$nH[i],maxit = gs_ANN$maxit[i],rang = gs_ANN$rang[i],silent = TRUE,MaxNWts = 10000)
  
  #Evaluate the model
  real <- max.col(val_target)
  pred <- max.col(predict(tmp_nnet,val_input))
  evaluation <- rbind(evaluation,cbind(real,pred))
  
  #Confusion Matrix
  ann_cfm <- matrix(0,nrow = 3, ncol = 3)
  ann_cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  ann_cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  ann_cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  ann_cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  ann_cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  ann_cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  ann_cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  ann_cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  ann_cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  ANN_result[i,1] <- gs_ANN$nH[i]
  ANN_result[i,2] <- gs_ANN$maxit[i]
  ANN_result[i,3] <- gs_ANN$rang[i]
  ANN_result[i,4:5] <- t(perf_eval_multi(ann_cfm))
}

#Check best and worst combination of ANN
best_ANN_result <- ANN_result[order(ANN_result[,5],decreasing = TRUE),]
colnames(best_ANN_result) <- c("nH","Maxit","rang","ACC","BCR")
best_ANN_result

#Train final Model
best_nH1 <- best_ANN_result[1,1]
best_maxit1 <- best_ANN_result[1,2]
best_rang1 <- best_ANN_result[1,3]

ptm <- proc.time()
ANN_final <- nnet(data_input,data_target,size = best_nH1, maxit = best_maxit1, rang = best_rang1, MaxNWts = 10000)
ANN.Time <- proc.time() - ptm
ANN.Time

ANN_pred <- predict(ANN_final, tst_input)
ANN_cfm1 <- table(max.col(tst_target), max.col(ANN_pred))
ANN_cfm1

perf_table[3,] <- perf_eval_multi(ANN_cfm1)
perf_table

#Question 2
#Bagging with CART

set.seed(12345)

#Train bagged model
library(ipred)
library(rpart)
library(mlbench)
library(caret)
library(party)

nbagg <- seq(from = 30, to = 300, by = 30)
Bagging_Result <- matrix(0,length(nbagg),3)

iter_cnt = 1

for(i in 1:length(nbagg)){
  cat("Bagging Training: the number of bootstrap:",nbagg[i],"\n")
  evaluation <- c()
  
  tmp_bagging <- cforest(GradeYN ~ ., data = CART_trn, controls = cforest_control(mincriterion = best_criterion, minsplit = best_split, maxdepth = best_depth,mtry = 0, ntree = nbagg[i]))
  
  real <- CART_val$GradeYN
  pred <- predict(tmp_bagging, newdata = CART_val)
  evaluation <- rbind(evaluation,cbind(real,pred))
  
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
  
  Bagging_Result[iter_cnt,1] = nbagg[i]
  Bagging_Result[iter_cnt,2:3] = t(perf_eval_multi(cfm))
  iter_cnt = iter_cnt + 1
}

Bagging_Result_order <- Bagging_Result[order(Bagging_Result[,3],decreasing = T),]
colnames(Bagging_Result_order) <- c("Bootstrap","ACC","BCR")
Bagging_Result_order

best_bootstrap <- Bagging_Result[1,1]

Bagging_Final <- cforest(GradeYN ~ ., data = CART_data, controls = cforest_control(mincriterion = best_criterion, minsplit = best_split, maxdepth = best_depth, mtry = 0, ntree = best_bootstrap))
Bagging_pred <- predict(Bagging_Final, newdata = CART_tst)

Bagging_cfm <- table(CART_tst$GradeYN, Bagging_pred)
Bagging_cfm

perf_table[4,] <- perf_eval_multi(Bagging_cfm)
perf_table

#Question 3
#Random Forest
library(randomForest)
ntree <- seq(from = 30, to = 300, by = 30)

RF_Result <- matrix(0,length(ntree),3)
colnames(RF_Result) <- c("Tree","ACC","BCR")

iter_cnt = 1

for(i in 1:length(ntree)){
  cat("RandomForest Training:",ntree[i],"\n")
  tmp_RF <- randomForest(GradeYN ~., data = CART_trn, ntree = ntree[i], mincriterion = best_criterion, min_split = best_split, maxdepth = max_depth, importance = TRUE, do.trace = TRUE)
  
  RF.pred <- predict(tmp_RF, newdata = CART_val, type = "class")
  RF.cfm <- table(CART_val$GradeYN, RF.pred)
  print(tmp_RF)
  RF_Result[iter_cnt,1] = ntree[i]
  RF_Result[iter_cnt,2:3] = t(perf_eval_multi(RF.cfm))
  iter_cnt = iter_cnt + 1
}

RF_Result <- RF_Result[order(RF_Result[,3],decreasing = T),]

best_bootstrap2 <- RF_Result[1,1]

ptm <- proc.time()
RF_Final <- randomForest(GradeYN ~ ., data = CART_data, ntree = 30, importance = TRUE, do.trace = TRUE)
RF.Time <- ptm - proc.time()

print(RF_Final)
plot(RF_Final)

Var.imp <- importance(RF_Final)
impor <- Var.imp[order(Var.imp[,4],decreasing = TRUE),]
summary(Var.imp)
barplot(Var.imp[order(Var.imp[,4],decreasing = TRUE),4])

RF_pred <- predict(RF_Final, newdata = CART_tst, type = "class")
RF_cfm <- table(CART_tst$GradeYN, RF_pred)
RF_cfm

perf_table[5,] <- perf_eval_multi(RF_cfm)
perf_table


library(ggplot2)

Bagging_data <- Bagging_Result[order(Bagging_Result[,1],decreasing = F),]
No.Bootstrap <- Bagging_data[,1]
CART_Bagging_BCR <- Bagging_data[,3]
RF_BCR <- (RF_Result[order(RF_Result[,1],decreasing = F),])[,3]
BCR_summary <- data.frame(No.Bootstrap,CART_Bagging_BCR,RF_BCR)

GRAPH <- ggplot(data = BCR_summary)+geom_line(aes(x=No.Bootstrap,y=CART_Bagging_BCR))+geom_line(aes(x=No.Bootstrap,y=RF_BCR))
GRAPH
GRAPH + ylab("Value of BCR")

##Question 4
best_nH <- best_nH1
best_maxit <- best_maxit1
best_rang <- best_rang1

val_perf <- c()

iter_cnt = 1

for(i in 1:30){
  evaluation <- c()
  
  tmp_nnet <- nnet(data_input, data_target, size = best_nH, maxit = best_maxit, rang = best_rang, MaxNWts = 10000)
  
  real <- max.col(tst_target)
  pred <- max.col(predict(tmp_nnet,tst_input))
  evaluation <- rbind(evaluation, cbind(real,pred))
  
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
  
  val_perf <- rbind(val_perf,t(perf_eval_multi(Final_cm)))
}
val_perf
colnames(val_perf) <- c("ACC","BCR")

mean_ACC <- mean(val_perf[,1])
sd_ACC <- sd(val_perf[,1])
ANN_ACC <- cbind(mean_ACC,sd_ACC)
ANN_ACC

mean_BCR <- mean(val_perf[,2])
sd_BCR <- sd(val_perf[,2])
ANN_BCR <- cbind(mean_BCR,sd_BCR)
ANN_BCR
ANN_iter_summary <- data.frame(ANN_ACC,ANN_BCR)

#Question 5
#Bagging with Neural Network 
library(caret)
library(doParallel)

cl <- makeCluster(4)
registerDoParallel(cl)

nrepeats = seq(from = 30, to = 300, by = 30)
ann.bagging.result <- c()
summary.table <- matrix(0,10,5)
colnames(summary.table) <- c("Bootstrap","mean_ACC","sd_ACC","mean_BCR","sd_BCR")

gc()

for(i in 1:length(nrepeats)){
  cat("Training Bagging ANN: The Number of Bootstrap:",nrepeats[i],"\n")
  
  for(j in 1:10){
    cat("The Number of Repeats:",j,"\n")
    evaluation <- c()
    
    tmp_ann.bagging.model <- avNNet(data[,input_idx],data[,target_idx], size = best_nH1, maxit = best_maxit1, rang = best_rang1, repeats = nrepeats[i], bag = TRUE, trace = TRUE, MaxNWts = 10000)
    
    real <- max.col(tst_target)
    pred <- max.col(predict(tmp_ann.bagging.model,tst_input))
    evaluation <- rbind(evaluation, cbind(real,pred))
    
    Final_cm <- matrix(0,nrow = 3,ncol = 3)
    Final_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
    Final_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
    Final_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
    Final_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
    Final_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
    Final_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
    Final_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
    Final_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
    Final_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
    
    ann.bagging.result <- rbind(ann.bagging.result,t(perf_eval_multi(Final_cm)))
  }
  summary.table[i,1] <- nrepeats[i]
  summary.table[i,2] <- mean(ann.bagging.result[,1])
  summary.table[i,3] <- sd(ann.bagging.result[,1])
  summary.table[i,4] <- mean(ann.bagging.result[,2])
  summary.table[i,5] <- sd(ann.bagging.result[,2])
}
ann.bagging.result
summary.table


best_repeats <- 30

ann.bagging.best <- c()


evaluation <- c()
annb.model <- avNNet(data[,input_idx], data[,target_idx], size = 40 , maxit = 200, rang = 0.9, repeats = best_repeats, bag = TRUE, trace = TRUE, MaxNWts = 10000)

real <- max.col(tst_target)
pred <- max.col(predict(annb.model,tst_input))
evaluation <- rbind(evaluation, cbind(real,pred))

annb_cm <- matrix(0,nrow = 3, ncol = 3)
annb_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
annb_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
annb_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
annb_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
annb_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
annb_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
annb_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
annb_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
annb_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))

ann.bagging.best <- rbind(ann.bagging.best,t(perf_eval_multi(annb_cm)))
ann.bagging.result

perf_table[6,] <- perf_eval_multi(annb_cm)
perf_table

##Question 6
#Ada
library(ada)
library(adabag)
library(tidyverse)

gs <- list(iter = c(50,100,200),
           bag.frac = c(0.1,0.25,0.5)) %>%
  cross_df()

ada_perf <- data.frame()
gc()
iter_cnt = 1
for(i in 1:nrow(gs)){
  cat("Training adaboost: the number of population:",gs$iter[i],",ratio:",gs$bag.frac[i],"\n")
  evaluation <- c()
  
  tmp_iter <- gs[i,1]
  tmp_frac <- gs[i,2]
  
  tmp_adaboost <- boosting(GradeYN ~., data = CART_trn, boos = TRUE, mfinal = gs$iter[i], bag.frac = gs$bag.frac[i], control = rpart.control(mincriterion = 0.9, minsplit = 10))
  
  real <- CART_val$GradeYN
  pred <- predict(tmp_adaboost, CART_val[,input_idx])
  evaluation <- rbind(evaluation, cbind(real,pred$class))
  
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
  
  tmp_gs <- data.frame(tmp_iter, tmp_frac)
  tmp_ada <- t(perf_eval_multi(cfm))
  
  ada_perf <- rbind(ada_perf,cbind(tmp_gs,tmp_ada))
}
colnames(ada_perf) <- c("iteration","bag.frac","ACC","BCR")
ada_perf <- ada_perf[order(ada_perf[,4], decreasing = TRUE),]
ada_perf

best_iter <- ada_perf[1,1]
best_ration <- ada_perf[1,2]

ptm <- proc.time()
adaboost.model <- boosting(GradeYN ~., data = CART_data, boos = TRUE, iter = 50, bag.frac = 0.5)
Boosting.Time <- proc.time() - ptm
Boosting.Time

ada_print <- print(adaboost.model)

adaboost.pred <- predict(adaboost.model,CART_tst[,input_idx])
adaboost.cfm <- table(CART_tst$GradeYN, adaboost.pred$class)

perf_table[7,] <- perf_eval_multi(adaboost.cfm)
perf_table

##Question 7
#GBM 
library(gbm)
library(caret)

GBM.trn <- data.frame(trn[,input_idx],GradeYN = trn[,target_idx])
GBM.val <- data.frame(val[,input_idx],GradeYN = val[,target_idx])
GBM.tst <- data.frame(tst[,input_idx],GradeYN = tst[,target_idx])

gbmGrid <-  expand.grid(n.trees = c(400,500,600,700),
                        shrinkage = c(0.05, 0.1, 0.15, 0.2))

gbm_perf <- matrix(0,nrow(gbmGrid),4)
gc()
iter_cnt = 1
for(i in 1:nrow(gbmGrid)){
  cat("Training GBM: the number of population:",gbmGrid$n.trees[i],",shrinkage:",gbmGrid$shrinkage[i],"\n")
  evaluation <- c()
  
  tmp_gbm <- gbm.fit(GBM.trn[,input_idx],GBM.trn[,target_idx], distribution = "multinomial",verbose = TRUE, n.trees =gbmGrid$n.trees[i], shrinkage = gbmGrid$shrinkage[i])
  
  real <- GBM.val$GradeYN
  pred <- as.data.frame(predict(tmp_gbm, GBM.val[,input_idx], type = "response", n.trees = gbmGrid$n.trees[i]))
  pred <- max.col(pred)
  evaluation <- rbind(evaluation,cbind(real,pred))
  
  #Confusion Matrix
  gbm_cfm <- matrix(0,nrow = 3, ncol = 3)
  gbm_cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  gbm_cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  gbm_cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  gbm_cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  gbm_cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  gbm_cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  gbm_cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  gbm_cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  gbm_cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  gbm_perf[iter_cnt,1] <- gbmGrid$n.trees[i]
  gbm_perf[iter_cnt,2] <- gbmGrid$shrinkage[i]
  gbm_perf[iter_cnt,3:4] <- t(perf_eval_multi(gbm_cfm))
  iter_cnt = iter_cnt +1
}
gbm_perf <- gbm_perf[order(gbm_perf[,4],decreasing = TRUE),]
colnames(gbm_perf) <- c("n.trees","shrinkage","ACC","BCR")
write.csv(gbm_perf, file  = "parameter.csv", row.names = FALSE)

best_tree <- gbm_perf[1,1]
best_shrinkage <- gbm_perf[1,2]

ptm <- proc.time()
gbm.model <- gbm.fit(data[,input_idx],data[,target_idx], distribution = "multinomial",verbose = TRUE, n.trees = 700, shrinkage = 0.2)
gbm.Time <- proc.time() - ptm
gbm.Time

summary <- summary(gbm.model)

gbm.pred <- as.data.frame(predict(gbm.model, GBM.tst[,input_idx], type = "response", n.trees = 700))
gbm.cfm <- table(max.col(gbm.pred), GBM.tst$GradeYN)
gbm.cfm

perf_table[8,] <- perf_eval_multi(gbm.cfm)
perf_table


##Question 8
perf_table_final <- perf_table[order(perf_table[,2],decreasing = TRUE),]
perf_table_final



#Extra Question
perf_table.extra <- matrix(0,nrow = 1, ncol = 2)
colnames(perf_table.extra) <- c("ACC","BCR")
rownames(perf_table.extra) <- c("result")

#upample
library(caret)
Damage_up <- upSample(subset(Final_data, select = -Class),Final_data$Class)
table(Damage_up$Class)
trn <- Final_data[sample(nrow(Final_data),7430),]
val <- Final_data[sample(nrow(Final_data),2477),]
tst <- Final_data[sample(nrow(Final_data),3128),]
CART_trn_up <- data.frame(trn[,input_idx], GradeYN = trn[,target_idx])
CART_val_up <- data.frame(val[,input_idx], GradeYN = val[,target_idx])
CART_tst_up <- data.frame(tst[,input_idx], GradeYN = tst[,target_idx])
RF.ensem <- randomForest(GradeYN ~., data = CART_trn_up, ntree = 300, mincriterion = 0.9, min_split = 10, maxdepth = 0, importance = TRUE, do.trace = TRUE)
RF.ensem.pred <- predict(RF.ensem, newdata = CART_tst_up, type = "class")
RF.ensem.cfm <- table(CART_tst_up$GradeYN,RF.ensem.pred)
RF.ensem.cfm
perf_table.extra[1,] <- perf_eval_multi(RF.ensem.cfm)
perf_table.extra

print(RF.ensem)
plot(RF.ensem)

Var.imp.up <- importance(RF.ensem)
summary(Var.imp.up)
barplot(Var.imp.up[order(Var.imp.up[,4],decreasing = TRUE),4])

