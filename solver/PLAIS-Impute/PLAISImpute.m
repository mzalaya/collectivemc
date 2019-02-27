function [U, S, V, output ] = PLAISImpute( D, lambda, para, datatypes, n_cols, mask )

% ------- % 
% Inputs  %                                            %
% ------- % 

% D          : [m x n] sparse observed matrix
% lambda     : nuclear norm penalty
% para:
%    tol     :     convergence tolerance
%    maxIter : maximum number of iterations
%    decay   :   contral decay of lambda in each iteration.
%    maxR    :    maximum rank allowed during iteration
%    test    :    run test RMSE, see usage in TestRecsys.m

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.8;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

[row, col, data] = find(D);
[m, n] = size(D);

R = randn(n,maxR);
[U0, ~] = powerMethod(D, R, 5, 1e-6);

lambdaMax = topksvd(D, 1, 5);

[Ur, S, V0] = svd(U0'*D, 'econ'); 
%lambdaMax = max(vec_mat(S));
U0 = U0*(Ur*S); %HERE I changed

U1 = U0; 
V1 = V0;

a0 = 1;
a1 = 1;

maxIter = para.maxIter;
tol = para.tol;

lambdas = zeros(maxIter, 1);
obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RankIn = zeros(maxIter, 1);
RankOut = zeros(maxIter, 1);

t = tic;
for i = 1:maxIter
    lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
    lambdas(i) = lambdai;
    
    bi =  (a0 - 1)/a1;
    part0 = partXY(U0', V0', row, col, length(data)); 
    part1 = partXY(U1', V1', row, col, length(data));
    part = (1 + bi)*part1' - bi*part0';
    spa = sparse(row, col, part', m, n); 
    
    [~, spa_grad] = LikelihoodAndGradLikelihood(D, spa, datatypes, n_cols, mask);
    % [row_grad, col_grad, data_grad] = find(spa_grad);
    % setSval(spa_grad, data_grad, length(data_grad));
    
    R = filterBase(V1, V0, 1e-10 );
    R = R(:, 1:min([size(R, 2), maxR]));
    RankIn(i) = size(R, 2);
    
    pwTol = max(1e-6, lambdaMax * 0.95^i);
    [Q, pwIter] = powerMethodAccMatComp( U1, V1, U0, V0, spa_grad, bi, R, 10, pwTol);
    hZ = ((1+bi)*(Q'*U1))*V1' - (bi*(Q'*U0))*V0' + Q'*spa_grad;
    [Ui, S, Vi ] = SVT(hZ, lambdai);
    Ui = Q*(Ui*S);
        
    U0 = U1;
    U1 = Ui;
    
    V0 = V1;
    V1 = Vi;
    
    RankOut(i) = nnz(S);
    ai = (1e-3 + (sqrt(1e-3 + 4*a0^2)))/2; % 1e-3
    a0 = a1;
    a1 = ai;
    
    % vec_objVal  = partXY(Ui', Vi', row, col, length(data)); 
    vec_objVal  = partXY(Ui', Vi', row, col, length(data)); 
    spa_objVal = sparse(row, col, vec_objVal', m, n);
    [objVal, ~] = LikelihoodAndGradLikelihood(D, spa_objVal, datatypes, n_cols, mask);
    objVal = objVal + lambdai*sum(vec_mat(S));
    obj(i) = objVal;
   
    if(i > 1)
        delta = obj(i - 1) - obj(i);
        if(delta < 0)
             a0 = 1;
             a1 = 1;
        end
    else
        delta = inf;
    end
    fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.3d; power(iter %d, rank %d) \n', ...
        i, obj(i), delta, nnz(S), lambdai, pwIter, size(R, 2));
  
    % testing performance
%       if(isfield(para, 'test'))
%           tempS = partXY(Ui', Vi', row, col, length(data));
%           tempsS = sparse(row, col, tempS', m, n);
%           RMSE(i) = MatCompRMSE(D, tempsS);
%           fprintf('RMSE %.2d \n', RMSE(i));
%       end
     
    % checking covergence
    if(i > 1 && abs(delta) < tol)
        break;
    end
    if (objVal > 1e60)
        break;
    end
    
    Time(i) =  toc(t);

end

[U, S, V] = svd(U1, 'econ');
V = V1*V;
                            
output.obj = obj(1:i);
output.Rank = nnz(S);
output.RMSE = RMSE(1:i);
output.RankIn = RankIn(1:i);
output.RankOut = RankOut(1:i);
output.Time = Time(1:i);
output.lambdas = lambdas(1:i);
end
                            
function [vec_M] = vec_mat(M)
vec_M = M(:);
end