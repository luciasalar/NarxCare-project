require(scales)
require(ggplot2)
library(epitools) #package for odds ratio
library(dplyr)


##Task 1: ks test
#read result file, this result file contains the prediction probability and the  outcome
re <- read.csv("/Users/luciachen/Desktop/simulated_data/results/CVXR_unadjustedLG_datashift.csv")

#we obain the prediction probablity and outcome, combine them into table
re <- cbind(y_probs, test_y)
re <- as.data.frame(re)
colnames(re) <- c('prob', 'pred')
class_zero = re[re$pred ==0, ]
class_one = re[re$pred ==1, ]

#perform ks test score on the prediction results between two classes 
ks.test(class_zero$prob, class_one$prob)


#Task 2:  produce the plot that compares overdose and non-overdoes observations in each bin
#rescale prediction probabilty to a range of 0 - 990
rescaled_scores <- rescale(y_probs, to=c(0, 990))
summary(rescaled_scores)
# form a table (compare.csv) that contains the rescaled scores, prediction probability, outcome and Medd
compare <- data.frame(rescaled_scores, y_probs, test_y$Dx_OpioidOverdose_0to1_Y, test_X$Rx_Medd_2m)
colnames(compare) <- c('rescaled_scores', 'y_probs', 'Dx_OpioidOverdose_0to1_Y', 'Rx_Medd_2m')
write.csv(compare, "/home/groups/sherrir/luciachn/simulated_data/results/compare_scores.csv")


#Read compare.csv
compare <- read.csv("/Users/luciachen/Desktop/Desktop_MacbookPro/simulated_data/results/compare_scores.csv")
compare$rescaled_scores <- round(compare$rescaled_scores) #round up the scores

#bin the scores according to threshold, NarxCare defined these bins on the White Paper, this function returns the points bin
binned <- compare %>% mutate(points_bin = cut(rescaled_scores, breaks=c(0, 50, 100, 150, 200, 250,  300, 350, 400, 450, 500, 550, 600, 650, 700,750, 800, 850, 900, 950, 1000)))


#Here we plot overdose and non-overdose observations in each bin
#get sum of overdose observations in each bin
overdose_bins_cul <- aggregate(Dx_OpioidOverdose_0to1_Y ~ points_bin, binned, sum)

#to get the sum of non-overdose observation, we revert the coding, turn 1 to 0 and 0 to 1 then aggregate the count
binned$recoded_y <- -1 * (compare$Dx_OpioidOverdose_0to1_Y - 1) 
overdose_bins_cul_zero <- aggregate(recoded_y ~ points_bin, binned, sum)

#getting the percentage of overdose and non-overdose in each bin
overdose_bins_cul$percentage <- overdose_bins_cul$Dx_OpioidOverdose_0to1_Y / sum(overdose_bins_cul$Dx_OpioidOverdose_0to1_Y)
overdose_bins_cul_zero$percentage <- overdose_bins_cul_zero$recoded_y / sum(overdose_bins_cul_zero$recoded_y)

overdose_bins_cul$group <- 'overdose'
overdose_bins_cul_zero$group <-'non-overdose'

#put all the columns together to form a table for the plotting
overdose <- data.frame(overdose_bins_cul$percentage, overdose_bins_cul$points_bin, overdose_bins_cul$group)
colnames(overdose) <- c('percentage', 'points_bin', 'group')
non_overdose<- data.frame(overdose_bins_cul_zero$percentage, overdose_bins_cul_zero$points_bin, overdose_bins_cul_zero$group)
colnames(non_overdose) <- c('percentage', 'points_bin', 'group')

bar_chart_df <- rbind(overdose, non_overdose)

#plot the table
ggplot(bar_chart_df, aes(points_bin, percentage)) +   
  geom_bar(aes(fill = group), position = "dodge", stat="identity") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 0.5, size = 16), axis.text.y = element_text(size = 16), axis.title.y = element_text(size = 16))

# Task 3: odds ratio 

overdose_all <- cbind(overdose_bins_cul_zero, overdose_bins_cul)
overdose_raw <- overdose_all %>% dplyr::select('recoded_y','Dx_OpioidOverdose_0to1_Y')

calculate_odds_ratio <- function(row_number) {
  odds_ratio_bin <- as.matrix(overdose_raw[c(1,row_number),], nrow=2, ncol=2, byrow=TRUE)
  
  # the odds ratio is (a/b) / (c/d), we don't want the denominator to be 0 so we are adding 1 to each column to smooth the values,
  odds_ratio_bin[, 1] <- odds_ratio_bin[, 1] + 1
  odds_ratio_bin[, 2] <- odds_ratio_bin[, 2] + 1
  
  print(fisher.test(odds_ratio_bin))
  
}

#calculate odds ratio for all the bins
calculate_odds_ratio(2)
calculate_odds_ratio(3)
calculate_odds_ratio(4)
calculate_odds_ratio(5)
calculate_odds_ratio(6)
calculate_odds_ratio(7)
calculate_odds_ratio(8)
calculate_odds_ratio(9)
calculate_odds_ratio(10)
calculate_odds_ratio(11)
calculate_odds_ratio(12)
calculate_odds_ratio(13)
calculate_odds_ratio(14)
calculate_odds_ratio(15)


#Task 4
#Medd as criteria
summary(compare$Rx_Medd_2m)

#does it make sense to 
compare$rescaled_Rx_Medd_2m <- rescale(compare$Rx_Medd_2m, to=c(0, 43200))
compare$rescaled_Rx_Medd_2m <- round(compare$rescaled_Rx_Medd_2m)
summary(rescaled_Rx_Medd_2m)

binned2 <- compare %>% mutate(points_bin = cut(rescaled_Rx_Medd_2m, breaks=c(0, 1200, 3000, 5400, 7200, 9000,10800, 12600, 18000, 21600, 28800, 36000, 43200)))

Redd_bins_cul<- aggregate(Dx_OpioidOverdose_0to1_Y ~ points_bin, binned2, sum)




