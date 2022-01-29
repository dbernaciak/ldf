function [res_likelihoods, res_weights] = second_layer(loglikehood, deltas, is_argmax)

T = size(loglikehood,2);
res_weights = zeros(length(deltas), T, size(loglikehood,1));
res_likelihoods = zeros(length(deltas), T);

for i = 1:length(deltas)
    delta = deltas(i);
    weights = zeros(T, size(loglikehood,1));
    momentum = zeros(T, 1);
    cumsum = zeros(T, size(loglikehood,1));

    for t = 1:T
        if is_argmax
            p = zeros(size(loglikehood,1), 1);
            if t==1
                [~, am] = max(cumsum(t, :));
                p(am, :) = 1;
            else  
                [~, am] = max(cumsum(t-1, :));
                p(am, :) = 1;
            end
            weights(t, :) = p;
            momentum(t, :) = log(sum(exp(loglikehood(:, t)) .* p));
        else
            if t==1
                m = 0;
                p = exp(cumsum(t, :) - log(sum(exp(cumsum(t, :) - m))) - m);  
            else
                m = max(cumsum(t-1, :));
                p = exp(cumsum(t-1, :) - log(sum(exp(cumsum(t-1, :) - m))) - m); 
            end
            weights(t, :) = p;
            momentum(t, :) = log(sum(exp(loglikehood(:, t)).' .* p));
        end
            if t==1
                cumsum(t, :) = loglikehood(:, t);
            else
                cumsum(t, :) = delta * cumsum(t-1, :) + loglikehood(:, t).';
            end
    end
    res_likelihoods(i, :) = momentum;
    res_weights(i, :, :) = weights;
end

end