library(arules)
library(arulesViz)
library(wordcloud)
library(tidyverse)
library(colorspace)
library(iplots)

#check data
mooc_dataset <- read.csv("big_student_clear_third_version.csv")
str(mooc_dataset)
dim(mooc_dataset) 

#preperation data
#step 1
Institute <- mooc_dataset[,c(2)]
Course <- mooc_dataset[,c(3)]
#step2
Region <- gsub(" ","",mooc_dataset[,c(10)])
Degree <- gsub(" ","",mooc_dataset[,c(11)])
#step3
RawTransactions <- paste(Institute, Course, Region, Degree, sep = '_')
#step4
Transaction_ID <- mooc_dataset[,c(6)]
MOOC_transactions <- paste(Transaction_ID, RawTransactions, sep = ' ')
write.csv(MOOC_transactions, file = "MOOC_User_Course.csv", row.names = FALSE, quote = FALSE)
#step5
a<-read.csv("MOOC_User_Course.csv")
a
str(a)
dim(a)



#2
#Question2_1
MOOC_single_format <- read.transactions("MOOC_User_Course.csv", format = "single", 
                                        header = TRUE, cols = c(1,2), rm.duplicates = TRUE, skip = 1)

summary(MOOC_single_format)
str(MOOC_single_format)

#Question2_2
item_name <- itemLabels(MOOC_single_format)
item_count <- itemFrequency(MOOC_single_format)*nrow(MOOC_single_format)
col <- brewer.pal(10,"Paired")
wordcloud(words = item_name, freq = item_count, min.freq = 100, 
          scale = c(1,0.2), col = col,  random.order = FALSE)


#Question2_3

itemFrequencyPlot(MOOC_single_format, support = 0.01, cex.names = 0.6)
itemFrequencyPlot(MOOC_single_format, topN = 5, type = "absolute", cex.names = 0.8)



#3
#Question3_1
Support <- c(0.0005,  0.001, 0.0025, 0.005)
Confidence <- c(0.0005, 0.001, 0.005)


matrix_rules <- matrix(0,4,3)
rownames(matrix_rules) <- paste0("Support = ",Support)
colnames(matrix_rules) <- paste0("Confidence = ",Confidence)
matrix_rules

for(i in 1:4){
  for(j in 1:3){
    tmp_a <- Support[i]
    tmp_b <- Confidence[j]
    cat("Support:",Support[i],",Confidence:",Confidence[j],"\n")
    
    tmp_rule <- apriori(MOOC_single_format, parameter = list(support = tmp_a, confidence = tmp_b))
    tmp_rule <- data.frame(length(tmp_rule), tmp_a, tmp_b)
    
    tmp_cnt <- tmp_rule[,1]
    matrix_rules[i,j] <- tmp_cnt
  }
}
matrix_rules


#Question3_2
rules <- apriori(MOOC_single_format, parameter = list(support = 0.001, confidence = 0.05))
inspect(rules)
inspect(sort(rules, by = "support"))
inspect(sort(rules, by = "confidence"))
inspect(sort(rules, by = "lift"))
rules
str(rules)

rules_df <- DATAFRAME(rules)
rules_df$Perf_Mea_New <- rules_df$support * rules_df$confidence * rules_df$lift
rules_df <- rules_df[order(rules_df[,8],decreasing = T),]
rules_df

plot(rules, method="graph", cex= 0.7, edgeCol = grey(0.005), arrowSize = 0.7)
plot(rules, method="paracoord")


par(mar=c(1,1,1,1))
plot(rules, method = "graph")
plot(rules, method = "graph", engine = "htmlwidget")
rule_xy <- subset(rules, lhs %pin% c("MITx") & rhs %pin% c("MITx"))
inspect(rule_xy)

#EXTRA_QUESTION
#grouped
plot(rules, method = "grouped")

plot(rules, method = "graph", interactive = T)

# scatterplot
plot(rules, method = "scatterplot", measure = c("confidence", "lift"), shading = "support", engine = "htmlwidget")

# matrix
plot(rules, method="matrix")
plot(rules, method="matrix", engine = "htmlwidget")

# Plot grouped matrix method
plot(rules, method = "grouped")






