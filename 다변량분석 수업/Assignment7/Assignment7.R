
options("scipen"=10) #지수표기를 숫자로 변경

install.packages("arules")
install.packages("arulesViz")
install.packages("wordcloud")
install.packages("tidyverse")
install.packages("colorspace")
install.packages("iplots")
 
library(arules)
library(arulesViz)
library(wordcloud)
library(tidyverse)
library(colorspace)
library(iplots)

# read csv data
mooc_dataset <- read.csv("/Users/hyungmin/Documents/R/big_student_clear_third_version.csv")
str(mooc_dataset)
dim(mooc_dataset) 

## Q1
# 1단계 : 4개 변수명 변경
mooc_dataset <- mooc_dataset %>% 
                rename(
                  Institute = institute, 
                  Course = course_id,
                  Region = final_cc_cname_DI,
                  Degree = LoE_DI
                )

str(mooc_dataset)

# 2단계 : Region에 해당하는 공백 제거
unique(mooc_dataset$Region) # check unique val
mooc_dataset$Region<-gsub(" ", "", mooc_dataset$Region) # eliminating space
unique(mooc_dataset$Region) # check unique val

# 3단계 : 네 변수를 합쳐서 하나의 변수로 만드시오.
mooc_dataset$RawTransactions <- paste(mooc_dataset$Institute, mooc_dataset$Course, mooc_dataset$Region, mooc_dataset$Degree, sep = "_")


# 4단계 : Attach Transaction ID and RawTransactions
mooc_dataset$MOOC_transactions <- paste(mooc_dataset$userid_DI, mooc_dataset$RawTransactions, sep = " ")

# 5 단계 : write CSV
write.table(mooc_dataset$MOOC_transactions, 
            "MOOC_User_Course2.csv", 
            sep = ',',
            col.names = FALSE,
            row.names = FALSE)

## Q2. 
# 2-1: Make Transaction using (Single type)
MOOC_trans <- read.transactions("MOOC_User_Course2.csv", header = FALSE,
                                format = "single", cols = c(1,2), rm.duplicates=TRUE)
summary(MOOC_trans) # check transaction data summary


# 2-2 : Draw WordCloud
# Item inspection
itemName <- itemLabels(MOOC_trans)
itemCount <- itemFrequency(MOOC_trans)*nrow(MOOC_trans)

# check color sets
#install.packages("RColorBrewer")
library(RColorBrewer) # for checking and using Color Pallete
display.brewer.all()

# draw word cloud
col <- brewer.pal(12, "Paired")
wordcloud(words = itemName, freq = itemCount, min.freq = 100, scale = c(2,0.01), 
          col = col, random.order = FALSE)

# 2-3 : Draw itemFrequencyPlot
par(mar = c(10, 10, 2, 2)) # par(mar = c(bottom, left, top, right))
itemFrequencyPlot(MOOC_trans, support = 0.01, cex.names=0.8, ylim = c(0, 0.05))
itemFrequencyPlot(MOOC_trans, topN = 5, support = 0.01, cex.names=0.8, ylim = c(0, 0.05))

## Q3
# 3-1 : Generate Rules
support <- c(0.0005, 0.001, 0.005, 0.01, 0.03, 0.05)
confidence <-c(0, 0.0005, 0.001, 0.005, 0.05, 0.1)

# make empty matrix for data input
num_rules<- matrix(0, nrow = length(support), ncol = length(confidence))
rownames(num_rules)<-c("sup_0.0005", "sup_0.001", "sup_0.005", "sup_0.01", "sup_0.03","sup_0.05")
colnames(num_rules)<-c("con_0", "con_0.0005", "con_0.001", "con_0.005", "con_0.05", "con_0.1")
num_rules

# conduct grid search
for (i in 1:length(support)){
  for (j in 1:length(confidence)){
    cat("conducting support = ", support[i], ", confidence = ", confidence[j], "\n")
    
    # Rule generation by Apriori
    tmp_rules <- apriori(MOOC_trans, parameter=list(support=support[i], confidence=confidence[j]))

    # Insert number of rules
    num_rules[i,j] <- dim(tmp_rules@lhs)[1]
    
    cat("DONE", '\n')
  }
}

num_rules

# 3-2 : Generate Association Rule using support = 0.001, confidence = 0.05
# Rule generation by Apriori
rules3_2 <- apriori(MOOC_trans, parameter=list(support=0.001, confidence=0.05))

# Check the generated rules
inspect(rules3_2)

# List the first three rules with the highest support values
inspect(sort(rules3_2, by="support"))[1:5,]

# List the first three rules with the highest confidence values
inspect(sort(rules3_2, by="confidence"))[1:5,]

# List the first three rules with the highest lift values
inspect(sort(rules3_2, by="lift"))[1:5,]

# make dataframe out of inspection
inspect_df <- data.frame(inspect(rules3_2))
dim(inspect_df)
str(inspect_df)

# make new variable(utility_var) : support * confindence * lift
inspect_df$utility_var<-inspect_df$support*inspect_df$confidence*inspect_df$lift
inspect_df

# sort by utility_var
inspect_df <- inspect_df[order(inspect_df$utility_var, decreasing = TRUE),]
inspect_df[1:3,]

# plot using graph method
plot(rules3_2, method="graph", cex= 0.7, edgeCol = grey(0.005), arrowSize = 0.7)
plot(rules3_2, method="paracoord")

# check X -> Y ,  Y -> X cases using subset
rule_interest <- subset(rules3_2, lhs %pin% c("MITx") & rhs %pin% c("MITx"))
inspect(rule_interest)

## EXTRA QUESTION -----------
? arulesViz::plot
# methods : "scatterplot", "two-key plot", "matrix", "matrix3D", "mosaic", "doubledecker", "graph", "paracoord", "grouped", "iplots"

# used data
rules3_2

# Plot scatterplot method
plot(rules3_2, method = "scatterplot", measure = c("confidence", "lift"), shading = "support")
plot(rules3_2, method = "scatterplot", measure = c("confidence", "lift"), shading = "support", engine = "htmlwidget")

# Plot two-key plot method
plot(rules3_2, method = "two-key plot")
plot(rules3_2, method = "two-key plot", engine = "htmlwidget")

# Plot matrix method
plot(rules3_2, method="matrix")
plot(rules3_2, method="matrix", engine = "htmlwidget")

# Plot matrix3D method
plot(rules3_2, method="matrix3D")

# Plot grouped matrix method
plot(rules3_2, method = "grouped")
plot(rules3_2, method = "grouped", control=list(k=5))

# Plot graph method
plot(rules3_2, method = "graph")
plot(rules3_2, method = "graph", engine = "htmlwidget")

# Plot paracoord method
plot(rules3_2, method = "paracoord")

# Plot mosaic method with biggest lift
mosaicRule<-sort(rules3_2, by="lift")[1,]
inspect(mosaicRule)
plot(mosaicRule, method="mosaic", data = MOOC_trans)

# Plot mosaic method with smallest lift
mosaicRule<-sort(rules3_2, by="lift")[51,]
inspect(mosaicRule)
plot(mosaicRule, method="mosaic", data = MOOC_trans)







  