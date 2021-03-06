function [momentum, weights, param_weights] = ldf(agent_loglikelihoods, levels, discount_factors, activation_functions, c)
if ~exist('c','var')
      c = 10^(-20);
end    

alphas = discount_factors;
is_argmax = activation_functions(1) == "argmax";
[loglikehood ,prob] = first_layer(agent_loglikelihoods, alphas, is_argmax, c);

if levels == 2
    deltas = discount_factors;
    is_argmax = activation_functions(2) == "argmax";
    [momentum, param_weights] = second_layer(loglikehood, deltas, is_argmax);
    weights = zeros(length(deltas), size(agent_loglikelihoods,2), size(agent_loglikelihoods,1));

   for k = 1:length(deltas)
      for i = 1:size(param_weights,2)
        weights(k, i, :) = sum(squeeze(prob(i, :, :)) .* squeeze(param_weights(k, i, :)));
      end
   end

 else
    momentum = loglikehood;
    weights = prob;
    param_weights = NaN;
 end