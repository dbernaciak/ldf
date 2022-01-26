argmax <- function(x) {
  which.max(x)
}


first_layer <- function(agent_loglikelihoods, alphas, is_argmax, cs) {
  
  T = dim(agent_loglikelihoods)[2]
  n_agents = dim(agent_loglikelihoods)[1]
  pi_t = matrix(1, length(alphas), n_agents) / n_agents
  prob = array(0, dim=c(T, length(alphas), n_agents))
  loglikehood = matrix(0, length(alphas), T)
  
  for (t in 1:T) {
    for (j in 1:length(alphas)) {
      alpha = alphas[j]
      if (is_argmax) {
        pi_t[j, ] = alpha * pi_t[j, ]
        p = rep(0, length(pi_t[j, ]))
        p[argmax(pi_t[j, ])] = 1
        prob[t, j, ] = p
        loglikehood[j, t] = sum(agent_loglikelihoods[ , t] * p)
        pi_t[j, ] = pi_t[j, ] + agent_loglikelihoods[ , t]
      }
      else {
        pi_t[j, ] = (pi_t[j, ]^alpha + cs) / (sum(pi_t[j, ]^alpha) + cs)  # pi_t|t-1
        prob[t, j, ] = pi_t[j, ]
        loglikehood[j, t] = log(sum(exp(agent_loglikelihoods[ , t]) * pi_t[j, ]))
        pi_t[j, ] = pi_t[j, ] * exp(agent_loglikelihoods[ , t])
        pi_t[j, ] = pi_t[j, ] / sum(pi_t[j])  # pi_t|t
      }
    }
  }
  return(list("loglikehood"=loglikehood, "prob"=prob))
}

second_layer <- function(loglikehood, deltas, is_argmax) {
  T = dim(loglikehood)[2]
  res_weights = array(0, dim=c(length(deltas), T, dim(loglikehood)[1]))
  res_likelihoods = matrix(0, length(deltas), T)
  
  for (i in 1:length(deltas)) {
    delta = deltas[i]
    weights = matrix(0, T, dim(loglikehood)[1])
    momentum = rep(0,T)
    csum = matrix(0, T, dim(loglikehood)[1])
    for (t in 1:T) {
      if (is_argmax) {
        p = rep(0, dim(loglikehood)[1])
        if (t==1) {
          p[argmax(csum[t, ])] = 1
        }
        else {
          p[argmax(csum[t-1, ])] = 1
        }
        weights[t, ] = p
        momentum[t] = log(sum(exp(loglikehood[ , t]) * p))
      }
      else {
        if (t==1) {
          m = 0
          p = exp(csum[t, ] - log(sum(exp(csum[t, ] - m))) - m)
        }
        else {
          m = max(csum[t-1, ])
          p = exp(csum[t-1, ] - log(sum(exp(csum[t-1, ] - m))) - m)
        }
        weights[t, ] = p
        momentum[t] = log(sum(exp(loglikehood[ , t]) * p))
      }
      if (t==1) {
        csum[t, ] = loglikehood[ , t]
      }
      else {
        csum[t, ] = delta * csum[t-1, ] + loglikehood[ , t]
      }
    }
    res_likelihoods[i, ] = momentum
    res_weights[i, , ] = weights
  }
  return(list("res_likelihoods"=res_likelihoods, "res_weights"=res_weights))
}

ldf <- function(agent_loglikelihoods, levels, discount_factors, activation_functions, c=10^(-20)) {
  
  alphas = discount_factors
  is_argmax = activation_functions[1] == "argmax"
  res = first_layer(agent_loglikelihoods, alphas, is_argmax, c)
  loglikehood = res$loglikehood
  prob = res$prob
  
  if (levels == 2) {
    deltas = discount_factors
    is_argmax = activation_functions[2] == "argmax"
    res = second_layer(loglikehood, deltas, is_argmax)
    momentum = res$res_likelihoods
    param_weights = res$res_weights
    weights = array(0, dim=c(length(deltas), dim(forecasts_lik)[2], dim(forecasts_lik)[1]))
    for (k in 1:length(deltas)) {
      for (i in 1:dim(param_weights)[2]) {
        weights[k, i, ] = apply(t(prob[i, , ]) * param_weights[k, i, ],1,sum)
      }
    }
    list("logscores"=momentum , "weights"=weights, "params_weights"=param_weights)
  }
  else{
    list("logscores"=loglikehood , "weights"=prob, "params_weights"=NULL)
  }
  
}


forecasts_lik = t(matrix(c(
  c(-2.302585093, -4.605170186, -4.605170186, -1.609437912, -4.605170186, -1.771956842, -1.609437912, -1.386294361,
   -1.386294361, -2.302585093, -1.386294361),
  c(-0.916290732, -1.203972804, -1.609437912, -2.995732274, -4.605170186, -2.302585093, -4.828313737, -1.386294361,
   -1.609437912, -2.991548932, -0.801038303),
  c(-2.302585093, -3.688879454, -3.688879454, -1.897119985, -2.9, -2.302585093, -4.1, -1.1, -0.1, -1.4,
   -1.897119985),
  c(-4.605170186, -4.605170186, -4.605170186, -4.605170186, -4.605170186, -0.9, -4.605170186, -0.510825624,
   -0.287682072, -1.491654877, -1.897119985),
  c(-1.203972804, -1.203972804, -1.290984181, -2.302585093, -4.605170186, -1, -4.605170186, -1.049822124,
   -2.302585093, -2.995732274, -0.597837001)), nrow=11, ncol=5))
discount_factors = c(1, 0.5)

res = ldf(forecasts_lik, 1, discount_factors, c("softmax"))
stopifnot(max(abs(apply(res$logscores,1,sum) - c(-24.54868304, -21.48288988))) < 10^(-8))
res = ldf(forecasts_lik, 1, discount_factors, c("argmax"))
stopifnot(max(abs(apply(res$logscores,1,sum) - c(-24.57489777, -25.00121507))) < 10^(-8))
res = ldf(forecasts_lik, 2, discount_factors, c("softmax", "softmax"))
stopifnot(max(abs(apply(res$logscores,1,sum) - c(-22.13047414, -22.23379625))) < 10^(-8))
res = ldf(forecasts_lik, 2, discount_factors, c("argmax", "argmax"))
stopifnot(max(abs(apply(res$logscores,1,sum) - c(-25.87418075, - 24.27844848))) < 10^(-8))

