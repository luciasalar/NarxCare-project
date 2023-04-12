
#This script contains the functions that generate the simmulated dataset

#generate pareoto distribution 
#rpareto generates random deviates, location:vector of (positive) location parameters., shape

get_pareto_var <- function(location, shape, num){
  data <- rpareto(num, location, shape)
  #data[data < threshold] <- 0     #convert value lower than threshold to 0
  data <- round(data, digits = 0)
  
  # add some noise, the noise should be very small and doesn't affect the distribution
  #noise <- sample(x = 0:threshold, size  = number_of_cases, replace = TRUE)
  new_data <- data 
  return (new_data)
  
}



# 
# 
#generate a variable x that has a predefined (population) correlation with an existing variable 𝑌
#https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables
complement <- function(y, rho, x) {
  if (missing(x)) x <- rnorm(length(y)) # Optional: supply a default if `x` is not given, normal distribution, mean 0, sd 1
  y.perp <- residuals(lm(x ~ y))
  rho * sd(y.perp) * y + y.perp * sd(y) * sqrt(1 - rho^2)
}




#generate a variable with normal distribution
get_normal_distribution <- function(num, upper, lower, mean, sd) {
  
  data <- rtruncnorm(n=num, a=lower, b=upper, mean=mean, sd=sd)
  print(summary(data))
  return(data)
}


#generate two correlated variables with normal distribution
get_correlated_normal_var <- function(var1, pho, upper_var2, lower_var2, mean_var2, sd_var2, rescale_upper, rescale_lower){

  #generate variable with normal distribution
  #upper_var, lower_var2, mean_var2, sd_var2: upper/lower bound, mean, sd of normal distribution,
  var2 <- get_normal_distribution(number_of_cases, upper_var2, lower_var2, mean_var2, sd_var2)

  #generate correlated variables
  cor_var<- complement(var1, pho, var2)
  #rescale the variable
  cor_var <- rescale(cor_var, to = c(rescale_upper, rescale_lower))

  print(quantile(cor_var, c(.20,.50,.60,.70, .80, .90, .99)))
  print(summary(cor_var))
  print(cor.test(cor_var, var1))

  return (cor_var)

}


##Count of Emergency Department stops in the last N days
#This function adds random count to the 30 days variable to create a 730 days variable
#var1: 30 day variable; max: maximum value of the new variable (730 days variable); threshold: values bigger than threshold will have probability of.0000001 to be sampled as none 0, otherwise the probability to be non-zero is .005
add_counts <- function(var1, max, threshold){
  #add random values to vector
  add_val <- function(x) {
      #values bigger than threshold will have probability of.0000001 to be sampled as none 0, otherwise the probability to be non-zero is .005
      probs = c(.9, rep(.005, threshold), rep(.0000001, max-threshold))
      #values to be added to the 30days variable
      add <- sample(x = 0:max, prob = probs, size  = 1, replace = TRUE) 
      #adding values to the 30 days variable 
      new_va <- x + add
      return (new_va)
  }
  #apply the above procedures to every value in the vector
  new_variable <- unlist(lapply(var1, add_val))
  #checking the statistics
  print(cor(new_variable, var1))
  print(summary(new_variable))
  print(sd(new_variable))

  return (new_variable)


}


add_counts_total <- function(var1, max, first){
  #add random values to vector
  add_val <- function(x) {
    
    probs = c(.9, rep(.1, first), rep(.000001, max-first))
    
    add <- sample(x = 0:max, prob = probs, size  = 1, replace = TRUE) 
    #add <- sample(c(0, 1, 3, 5, 7, 9, 13, 16, 18, 20), size = 1, replace = TRUE, prob = probs)
    new_va <- x + add
    return (new_va)
    
  }
  
  new_variable <- unlist(lapply(var1, add_val))
  print(cor(new_variable, var1))
  print(summary(new_variable))
  print(sd(new_variable))
  
  return (new_variable)
  
  
}






#getting two correlated variables with pareoto distribution for tobacco cession
get_pareto_var_Toba <- function(location, shape, num){

  data <- rpareto(num, location, shape)
  #data[data < thres] <- 0     #convert value lower than threshold to 0
  data <- round(data, digits = 0)

  # add some noise, the noise should be very small and doesnt affect the distribution
  noise <- sample(x = c(0, 1, 3, 6, 8, 20),  size  = number_of_cases, replace = TRUE, prob = c(0.995, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001))
  new_data <- data + noise
  return (new_data)
}


#Count of Home Based Primary Care (HBPC) stops in the last N days
#getting the referenced variable
get_pareto_var_HBPC <- function(location, shape, num){

  data <- rpareto(num, location, shape)
  data[data < 2] <- 0     #convert value lower than threshold to 0
  data <- round(data, digits = 0)
  # add some noise, the noise should be very small and doesnt affect the distribution
  noise <- sample(x = c(0, 1, 3, 6, 8),  size  = number_of_cases, replace = TRUE, prob = c(0.999, 0.0001, 0.0001, 0.0001, 0.0001))
  new_data <- data + noise
  return (new_data)
}




add_Medd <- function(var1, probs, max_rand){
  #add random values to vector
  add_val <- function(x) {

    random_add <- runif(1, min=0, max=max_rand)
    add <- sample(c(0, random_add), size = 1, replace = TRUE, prob = probs)
    new_va <- x + add
    return (new_va)

  }


  #add zeros to vector
  reduce <- function(x) {
    random_reduce <- runif(1, min=0, max=3000)
    reduce <- sample(c(0, random_reduce), size = 1, replace = TRUE, prob = probs)
    new_va <- x - reduce
    if (new_va < 0){
      return (0)
    }
    else {return (new_va)}
  }

  new_variable <- unlist(lapply(var1, add_val))
  new_variable <- unlist(lapply(new_variable, reduce))
  print(summary(new_variable))
  print(cor.test(var1, new_variable))


  return (new_variable)


}




