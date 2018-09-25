function [likelihood, grad_likelihood] = ...
    LikelihoodAndGradLikelihood(X_collective, W_params, datatypes, n_cols)

% Inputs:
[nb_rows, nb_cols] = size(X_collective);

likelihoods = [];
grads = {};

X_sources = {X_collective(:, 1:n_cols(1))};
W_sources = {W_params(:, 1:n_cols(1))};

k = 1;
while k <= size(n_cols, 2)-1
    X_sources{end+1} = X_collective(:, n_cols(k)+1:n_cols(k) + n_cols(k+1));
    W_sources{end+1} = W_params(:, n_cols(k)+1:n_cols(k) + n_cols(k+1));
    k = k + 1;
end

v = 1;
while v <= size(n_cols, 2)
    if strcmpi(datatypes{v}{1}, 'gaussian')
        link_function = @(eta) (0.5 * datatypes{v}{2}(2)^2 * eta.^2);
        link_function_prime = @(eta) (datatypes{v}{2}(2)^2 * eta);
    elseif strcmpi(datatypes{v}{1}, 'bernoulli')
        link_function = @(eta) (log(1. + exp(eta)));
        link_function_prime = @(eta) (exp(eta) ./ (1. + exp(eta)));
    elseif strcmpi(datatypes{v}{1}, 'binomial')
        link_function = @(eta) (datatypes{v}{2}(1) * log(1 + exp(eta)));
        link_function_prime = @(eta) (datatypes{v}{2}(1) * exp(eta) ./ (1. + exp(eta)));
    elseif strcmpi(datatypes{v}{1}, 'poisson')
        link_function = @(eta) (exp(eta));
        link_function_prime = @(eta) (exp(eta));
    elseif strcmpi(datatypes{v}{1}, 'exponential')
        link_function = @(eta) (- log(-eta));
        link_function_prime = @(eta) (- 1.0 ./ eta);
    else
        fprintf('Error! You must give the corresponding datatype of the vth %d source\n');
    end 

    % likelihood 
    temp_mat = link_function(W_sources{v});
    likelihood = X_sources{v} .* W_sources{v} - temp_mat;
    likelihood = sum(likelihood(:));
    likelihoods(end+1) = likelihood;

    % grad
    temp_mat_prime = link_function_prime(W_sources{v});
    grad = X_sources{v} - temp_mat_prime;
    grads{end+1} = grad;
    v = v + 1;
end 

likelihood =  (-1.0 /(nb_rows * nb_cols)) * sum(likelihoods);


i = 2;
grad_likelihood  = grads{1};
while i <= length(grads) 
    grad_likelihood = horzcat(grad_likelihood, grads{i});
    i = i+1;
end
grad_likelihood = (-1.0 /(nb_rows * nb_cols)) * grad_likelihood;
% 
grad_likelihood = sparse(grad_likelihood);
end
