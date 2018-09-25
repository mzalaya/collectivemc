function [source, mask, sources, masks, U_blocs, V_blocs] = ...
    CollectMCsimu(n_rows, n_cols, ranks, probas, datatypes)

% Inputs:
% datatypes = {{'gaussian', [1.,  0.5]}, {'poisson', loc=2.}, {'bernoulli', p=0.3}};
% datatypes = {{'gaussian', [1.,  0.5]}, {'exponential', loc=2.}, {'binomial', [N= 10, p=0.3]}};

sources = {};
masks = {}; 
U_blocs = {}; 
V_blocs = {}; 
v = 1;
while (v <= size(n_cols, 2))
    if strcmpi(datatypes{v}{1}, 'gaussian')
        randomness = @(d,r) (normrnd(datatypes{v}{2}(1), datatypes{v}{2}(2), [d,r]));
    elseif strcmpi(datatypes{v}{1}, 'bernoulli')
        randomness = @(d,r) (binornd(1, datatypes{v}{2}, [d,r]));
    elseif strcmpi(datatypes{v}{1}, 'binomial')
        randomness = @(d,r) (binornd(datatypes{v}{2}(1), datatypes{v}{2}(2), [d,r]));
    elseif strcmpi(datatypes{v}{1}, 'poisson')
        randomness = @(d,r) (poissrnd(datatypes{v}{2}, [d,r]));
    elseif datatypes{v}{1} == 'exponential'
        randomness = @(d,r) (exprnd(datatypes{v}{2}, [d,r]));
    else
        fprintf('Error, you must give the datatype corresponding', '\n');
    end 

    U = randomness(n_rows, ranks(v));
    V = randomness(n_cols(v), ranks(v));
    mask = rand(n_rows, n_cols(v)) <= probas(v);
    X = U * V';
    X = X .* mask;
    U_blocs{end+1} = U;
    V_blocs{end+1} = V;
    masks{end+1} = mask;
    sources{end+1} = X;

    v = v + 1;
end

i = 2;
source = sources{1};
while i <= length(sources) 
    source = horzcat(source, sources{i});
    i = i+1;
end 

j = 2;
mask = masks{1};
while j <= length(masks) 
    mask = horzcat(mask, masks{j});
    j = j+1;
end 
end
