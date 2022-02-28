install.packages("Epi")
library(tidyverse)
library(corrplot)
library(moments)
library(ggplot2)
library(Epi)
perf_eval2 <- function(cm){
  
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

perf_mat <- matrix(0, 1, 6)
colnames(perf_mat) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat) <- "Logstic Regression"

voice <- read.csv("voice.csv")

voice$label <- ifelse(voice$label=="male",1,0)

input_idx <- c(1:20)
target_idx <- 21


#data preperation
voice_input <- voice[,input_idx]
voice_target <- as.factor(voice[,target_idx])
voice_data <- data.frame(voice_input, voice_target)

str(voice_data)

# Conduct the normalization
voice_input <- voice[,input_idx]
voice_input <- scale(voice_input, center = TRUE, scale = TRUE)
voice_target <- voice[,target_idx]
voice_data <- data.frame(voice_input, voice_target)
#data analysis
for(i in 20:1){
  qqnorm(voice_data[,i])
  qqline(voice_data[,i])
  boxplot(main=i,
          voice_data[,i])
  m <- mean(voice_data[,i])
  sd <- sd(voice_data[,i])
  sk <- skewness(voice_data[,i])
  ku <- kurtosis(voice_data[,i])
  cat(sprintf(" %i 번 칼럼\n Mean: %f\n Standard deviation: %f\n Skewness: %f\n Kurtosis: %f\n", i, m, sd, sk, ku))
}


#이상치 제거
for (i in 1:20) {
  Q1 = quantile(voice_data[,i], probs = c(0.25),na.rm = TRUE)
  Q3 = quantile(voice_data[,i], probs = c(0.75),na.rm = TRUE)
  IQR = Q3 - Q1
  
  LC = Q1 - 1.5 * IQR
  UC = Q3 + 1.5 * IQR
  
  voice_data = subset(voice_data, voice_data[,i] >= LC & voice_data[,i] <= UC)
}

for(i in 20:1){
   boxplot(main=i,
          voice_data[,i])
 }




#상관관계
plot(voice_data[,1:20])
cor(voice_data[,1:20])
x<-cor(voice_data[,1:20])
corrplot(x)

input_idx <- c(1:20)
target_idx <- 21






# Split the data into the training/validation sets
set.seed(12345)
trn_idx <- sample(1:nrow(voice_data), round(0.7*nrow(voice_data)))
voice_trn <- voice_data[trn_idx,]
voice_tst <- voice_data[-trn_idx,]

# Train the Logistic Regression Model with all variables
full_lr <- glm(voice_target ~ ., family=binomial, voice_trn)
summary(full_lr)

lr_response <- predict(full_lr, type = "response", newdata = voice_tst)
lr_target <- voice_tst$voice_target
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >= 0.5)] <- 1
cm_full <- table(lr_target, lr_predicted)
cm_full

perf_mat[1,] <- perf_eval2(cm_full)
perf_mat


#패키지를 이용한 AUROC
t <-predict(full_lr,voice_tst)
head(t)
ROC(test=t, stat=voice_tst$voice_target, plot='ROC', AUC=T, main="testset ROC")



#AUROC
#lr_response를 기준으로 내림차순 시켜 계산하기 편하게 만들기
Roc1 <- data.frame(lr_response, voice_tst$voice_target)
Roc2<- arrange(Roc1, desc(lr_response),voice_tst$voice_target)
colnames(Roc2) <- c("P(Positive)", "Male")
Roc2

#tpr과 fpr값을 지정해주고
TPR1 <- length(which(Roc2$Male==1))
FPR1 <- length(which(Roc2$Male==0))

TPR_FPR <- cbind(0,0)
colnames(TPR_FPR) <- c("TPR","FPR")

TPR2 = 0 
FPR2 = 0


for(i in 1:nrow(Roc2)){
  if(Roc2[i,2]==1){
    TPR2 <- TPR2 + 1
  }else{
    FPR2 <- FPR2 + 1
  }
  TPR_tmp <- TPR2/TPR1
  FPR_tmp <- FPR2/FPR1
  TPR_FPR_tmp <- data.frame(TPR_tmp,FPR_tmp)
  colnames(TPR_FPR_tmp) <- c("TPR","FPR")
  TPR_FPR <- rbind(TPR_FPR,TPR_FPR_tmp)
}
TPR_FPR

#Ready to make ROC table
z <- c("0","0")
Roc3 <- rbind(z,Roc2)
Roc3

#Integrate table and Values for TPR and FPR
ROC_table <- data.frame(Roc3,TPR_FPR)
colnames(ROC_table) <- c("P(Positive)","Chance of Admit","True_Positive(Sensitivity)","False_Positive(1-Specificity)")
ROC_table

#Plot ROC Curve
ggplot(data = ROC_table, aes(x=`False_Positive(1-Specificity)`,y=`True_Positive(Sensitivity)`))+geom_line()+geom_abline(color = "red", linetype = "dashed")

#Calculate AUROC(Area Under ROC Curve)
TPR_FPR %>%
  arrange(FPR) %>%
  mutate(area_rectangle = (lead(FPR)-FPR)*pmin(TPR,lead(TPR)),
         area_triangle = 0.5 * (lead(FPR)-FPR)*abs(TPR-lead(TPR))) %>%
  summarise(area = sum(area_rectangle + area_triangle, na.rm = TRUE))


#---------------------------------------------------------------------------


voice_data<-voice_data[,-c(3,4,8,12,19)]


# Split the data into the training/validation sets
set.seed(12345)
trn_idx <- sample(1:nrow(voice_data), round(0.7*nrow(voice_data)))
voice_trn <- voice_data[trn_idx,]
voice_tst <- voice_data[-trn_idx,]

# Train the Logistic Regression Model with all variables
full_lr <- glm(voice_target ~ ., family=binomial, voice_trn)
summary(full_lr)

lr_response <- predict(full_lr, type = "response", newdata = voice_tst)
lr_target <- voice_tst$voice_target
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >= 0.5)] <- 1
cm_full <- table(lr_target, lr_predicted)
cm_full

perf_mat[1,] <- perf_eval2(cm_full)
perf_mat

#AUROC
t <-predict(full_lr,voice_trn)
head(t)
ROC(test=t, stat=voice_trn$voice_target, plot='ROC', AUC=T, main="trainset ROC")
