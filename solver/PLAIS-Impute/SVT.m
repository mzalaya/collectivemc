function [ U, S, V ] = SVT( Z, lambda, rnk )
%% ------------------------------------------------------------------------
% exact solve low rank proximal step
% (1/2)*|X - Z|_F^2 + lambda |X|_theta
%% ------------------------------------------------------------------------
%  regtype = 1: Capped L1 regularizer 
%            2: Log Sum Penalty
%            3: TNN
%% ------------------------------------------------------------------------

if(exist('rnk', 'var'))
    [U, S, V] = lansvd(Z, rnk, 'L');
else
    [U, S, V] = svd(Z, 'econ');
end
s = diag(S);
s = s - lambda;
s = s(s > 0);
svs = length(s);

U = U(:,1:svs);
V = V(:,1:svs);
S = diag(s);

end
