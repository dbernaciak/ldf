function [loglikehood, prob] = first_layer(agent_loglikelihoods, alphas, is_argmax, c)
    T = size(agent_loglikelihoods,2);
    n_agents = size(agent_loglikelihoods,1);
    pi_t = ones(length(alphas), n_agents) / n_agents;
    prob = zeros(T, length(alphas), n_agents);
    loglikehood = zeros(length(alphas), T);

    for t = 1:T
        for j = 1:length(alphas)
            alpha = alphas(j);
            if is_argmax
                pi_t(j,:) = alpha * pi_t(j,:);
                p = zeros(length(pi_t(j,:)), 1);
                [~, am] = max(pi_t(j,:));
                p(am, :) = 1;
                prob(t, j, :) = p;
                loglikehood(j, t) = sum(agent_loglikelihoods(:, t) .* p);
                pi_t(j, :) = pi_t(j, :).' + agent_loglikelihoods(:, t);
            else
                pi_t(j, :) = (pi_t(j, :) .^ alpha + c) / (sum(pi_t(j, :) .^ alpha) + c);
                prob(t, j, :) = pi_t(j, :);
                loglikehood(j, t) = log(sum(exp(agent_loglikelihoods(:, t)).' .* pi_t(j, :)));
                pi_t(j, :) = pi_t(j, :) .* exp(agent_loglikelihoods(:, t)).';
                pi_t(j, :) = pi_t(j, :) / sum(pi_t(j, :));
            end
        end
    end
end