
#This script contains the functions that generate the simmulated dataset

#generate pareoto distribution 
#rpareto generates random deviates, location:vector of (positive) location parameters., shape

get_pareto_var <- function(location, shape, num){
  data <- rpareto(num, location, shape)
  #data[data < threshold] <- 0     #convert value lower than threshold to 0
  data <- round(data, digits = 0)
  
  # add some noise, the noise should be very small and doesn't affect the distribution
  noise <- sample(x = c(0, 1, 3, 6, 8),  size  = num_of_obs, replace = TRUE, prob = c(0.995, 0.0001, 0.0001, 0.0001, 0.0001))
  new_data <- data + noise
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
  var2 <- get_normal_distribution(num_of_obs, upper_var2, lower_var2, mean_var2, sd_var2)

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

#reduce counts from the 730 days variable to create the 30 days variable
#reduce count: flip  N observations to 0, 

#convert anything lower than threshold to be a random number from 0 to threshold, reduce other values by a random number 0 to threshold
reduce_counts <- function(var1, maxi, threshold){
  #add random values to vector
  add_val <- function(x) {
    #values bigger than threshold will have probability of.0000001 to be sampled as non-zero, otherwise the probability to be non-zero is .005
    # rep() replicate a number N times
    probs = c(.6, rep(0.8, maxi))
    #sample a random value to be reduce from the 730days variable , the prob of 0 is .9, then
    add <- sample(x = 0:maxi, prob = probs, size = 1, replace = TRUE) 
    #adding values to the 30 days variable 
    new_va <- x - add
    
    if (new_va < threshold & new_va >= 0){
      return (new_va)
    }
    if (new_va >= threshold){ #if new num > threshold, convert to a random num < thres
      return (round(runif(1, min=0, max=threshold), digits = 0))
    }
    else{
      return (0)
    }
  }
  #apply the above procedures to every value in the vector
  new_variable <- unlist(lapply(var1, add_val))
  #checking the statistics
  print(summary(new_variable))
  print(sd(new_variable))
  
  return (new_variable)
  
  
}


#convert anything lower than threshold to be a random number from 0 to threshold, reduce other values by a random number 0 to threshold, threshold is the maximum value of the 30day variable, reduce 60% of the values with a random number sample from 0 to maxi of the 30day var, replace rest of the value by 0, if var > maxi of the 30 day var, replace it with a random number sample from 0 to maxi of 30day
reduce_counts_appointment <- function(var1, maxi, threshold){
  #add random values to vector
  add_val <- function(x) {
    #values bigger than threshold will have probability of.0000001 to be sampled as non-zero, otherwise the probability to be non-zero is .005
    # rep() replicate a number N times
    probs = c(.2, rep(1/maxi, maxi)) #40% are 0, rest is a random value, each has a equal chance being drawn
    #sample a random value to be reduce from the 730days variable , the prob of 0 is .9, then
    add <- sample(x = 0:maxi, prob = probs, size = 1, replace = TRUE) 
    #adding values to the 30 days variable 
    new_va <- x - add
    
    if (new_va < threshold & new_va >= 0){
      return (new_va)
    }
    if (new_va >= threshold){ #if new num > threshold, convert to a random num < thres
      return (round(runif(1, min=0, max=threshold), digits = 0))
    }
    else{
      return (0)
    }
  }
  #apply the above procedures to every value in the vector
  new_variable <- unlist(lapply(var1, add_val))
  #checking the statistics
  print(summary(new_variable))
  print(sd(new_variable))
  
  return (new_variable)
  
  
}




add_counts_total <- function(var1, max, first){
  #add random values to vector
  add_val <- function(x) {
    ##values bigger than threshold will have probability of.0000001 to be sampled as none 0, otherwise the probability to be non-zero is .005
    #control the prob of having outliers
    probs = c(.9, rep(.1, first), rep(.000001, max-first))
    
    add <- sample(x = 0:max, prob = probs, size  = 1, replace = TRUE) 
    #add <- sample(c(0, 1, 3, 5, 7, 9, 13, 16, 18, 20), size = 1, replace = TRUE, prob = probs)
    new_va <- x + add
    return (new_va)
    
  }
  
  new_variable <- unlist(lapply(var1, add_val))
 # print(cor(new_variable, var1))
 # print(summary(new_variable))
  #print(sd(new_variable))
  
  return (new_variable)
  
  
}




#Count of appointments  in the last N days
#getting the referenced variable
get_pareto_var_app <- function(location, shape, num){

  data <- rpareto(num, location, shape)
  data[data < 3] <- 0
  data <- round(data, digits = 0)
  # add some noise, the noise should be very small and doesnt affect the distribution
  noise <- sample(x = c(0, 1, 3, 6, 8),  size  = num_of_obs, replace = TRUE, prob = c(0.999, 0.0001, 0.0001, 0.0001, 0.0001))
  new_data <- data + noise
  return (new_data)
}


#adding random zero to the vector
add_random_zero <- function(var1){
  #add zeros to vector
  add_zero <- function(x) {
    add <- sample(c(0, 1), size = 1, replace = TRUE, prob = c(0.995, 0.005))
    new_va <- x * add
    return (new_va)
  }
  
  new_variable <- unlist(lapply(var1, add_zero))
  print(summary(new_variable))
  return (new_variable)
  
}

#adding random zero to the vector
add_random_zero_hospital <- function(var1){
  #add zeros to vector
  add_zero <- function(x) {
    add <- sample(c(0, 1), size = 1, replace = TRUE, prob = c(0.99, 0.01))
    new_va <- x * add
    return (new_va)
  }
  
  new_variable <- unlist(lapply(var1, add_zero))
  print(summary(new_variable))
  return (new_variable)
  
}



add_random_zero_appointment <- function(var1){
  #add zeros to vector
  add_zero <- function(x) {
    add <- sample(c(0, 1), size = 1, replace = TRUE, prob = c(0.60, 0.4))
    new_va <- x * add #flip 60% of numbers to zero
    if (new_va == 0) { #convert those 0 to a random number from 1- 40 
      new_va <- round(runif(1, 0, 40), digit = 0)
    }
    else{
      return (new_va)
    }
    return (new_va)
  }
  
  new_variable <- unlist(lapply(var1, add_zero))
  print(summary(new_variable))
  return (new_variable)
  
}




add_Medd <- function(var1, probs, max_rand){
  #add random values to vector
  add_val <- function(x) {
    
    #generate a random number
    random_add <- runif(1, min=-5000, max=max_rand)
    #sample a number according to probabilities 
    add <- sample(c(0, random_add), size = 1, replace = TRUE, prob = probs)
    #adding the new number to the original value
    new_va <- x + add
    
    if (new_va < 0){
      return (0)
    }
    return (new_va)

  }

  new_variable <- unlist(lapply(var1, add_val))
  #new_variable <- unlist(lapply(new_variable, reduce))
  print(summary(new_variable))
  print(cor.test(var1, new_variable))


  return (new_variable)


}




