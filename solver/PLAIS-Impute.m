function [U, S, V, output ] = PLAIS-Impute( D, lambda, para, datatypes, n_cols )

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

% ------- % 
% Outputs  %                                            %
% ------- % 
% U     :
% S     : 
% V     :
% Output:

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.3;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

maxIter = para.maxIter;
tol = para.tol;


lambdas = zeros(maxIter, 1);
obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RankIn = zeros(maxIter, 1);
RankOut = zeros(maxIter, 1);
[row, col, data] = find(D);
[m, n] = size(D);

%D = sparse(D);
R = randn(n, maxR);


[U0, ~] = powerMethod(D, R, 5, 1e-4);
[R, S, V0] = svd(U0'*D, 'econ'); %'econ'
D = sparse(D);

lambdaMax = max(vec_mat(S));
% lambdaMax = topksvd(D, 1, 5);
disp(lambdaMax);

V1 = V0;
U0 = U0 *(R*S);
U1 = U0; 

a0 = 1;
a1 = 1;
data_spa = data;
spa = sparse(row, col, data_spa, m, n);
 t = tic;
 
for i = 1:maxIter
   
    lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
    lambdas(i) = lambdai;
    bi =  (a0 - 1)/(a1+1); 
    
    part0 = partXY(U0', V0', row, col, length(data)); 
    part1 = partXY(U1', V1', row, col, length(data));
    part = (1 + bi)*part1' - bi*part0';
   
    spa_part = sparse(row, col, part', m, n);   
    
    [~, spa_grad] = LikelihoodAndGradLikelihood(D, spa_part, datatypes, n_cols);
    
    [row_grad, col_grad, data_grad] = find(spa_grad);
    
    spa = spa_grad;
    
    R = filterBase(V1, V0, 1e-6 ); %3000 1e-6
    
    R = R(:, 1:min([size(R, 2), maxR]));
    RankIn(i) = size(R, 2);
    
    
    [Q, pwIter] = powerMethodAccMatCompBis( U1, V1, U0, V0, spa, bi, R, 10, 1e-6);
    hZ = ((1+bi)*(Q'*U1))*V1' - (bi*(Q'*U0))*V0' + Q'*spa;
    [ Ui, S, Vi ] = SVTFastProx(hZ, lambdai);
    Ui = Q*(Ui*S); 
    
%    [ Ui, S, Vi ] = accExactSVD_APGnc( U1, V1, U0, V0, spa, bi, size(R, 2));
%    [Ui, S, Vi] = matcompSVD( U1, V1, U0, V0, spa, bi, size(R, 2));
%     S = diag(S);
%     nnzS = sum(S > lambdai);
%     Ui = Ui(:, 1:nnzS);
%     Vi = Vi(:, 1:nnzS);
%     S = S(1:nnzS);
%     S = S - lambdai;
%     S = diag(S);
%     Ui = Ui*S;
%     pwIter = inf;
    
    U0 = U1;
    U1 = Ui;
    
    V0 = V1;
    V1 = Vi;
    
    RankOut(i) = nnz(S);
    ai = (1 + sqrt(1 + 4*a0^2))/2;
    a0 = a1;
    a1 = ai;
    
    %vec_objVal  = partXY(Ui', Vi', row, col, length(data)); 
    %spa_objVal = sparse(row, col, vec_objVal', m, n);
    
    vec_objVal = partXY(Ui', Vi', row_grad, col_grad, length(data_grad));
    spa_objVal = sparse(row_grad, col_grad, vec_objVal', m, n);
    [objVal, ~] = LikelihoodAndGradLikelihood(D, spa_objVal, datatypes, n_cols);
    objVal = objVal + lambda*sum(vec_mat(S));
    obj(i) = objVal;
   
    
    if(i > 1)
        delta = obj(i - 1) - obj(i);
        if(delta < tol)
             a0 = 1;
             a1 = 1;
        end
    else
        delta = inf;
    end
    fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.3d; power(iter %d, rank %d) \n', ...
        i, obj(i), delta, nnz(S), lambdai, pwIter, size(R, 2));
  
    % testing performance
%     if(isfield(para, 'test'))
%         tempS = eye(size(U1, 2), size(V1, 2));
%         RMSE(i) = MatCompRMSE(U1, V1, tempS, ...
%             para.test.row, para.test.col, para.test.data);
%         fprintf('RMSE %.2d \n', RMSE(i));
%     end
      if para.test.test == true
          tempS = partXY(Ui', Vi', row, col, length(data));
          tempsS = sparse(row, col, tempS', m, n);
          RMSE(i) = MatCompRMSERMSE(D, tempsS);
          % tempS = eye(size(U1, 2), size(V1, 2));
          %RMSE(i) = MatCompRMSERMSE(D, Ui*Vi');
%           tempS = eye(size(U1, 2), size(V1, 2));
%           RMSE(i)= MatCompRMSE(U1, V1, tempS, row, col, data);
           fprintf('RMSE %.2d \n', RMSE(i));
      end
     
    % checking covergence
    if(i > 1 && abs(delta) < tol)
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

%--------------------------------------------%
function [U, S, V] = SVTFastProx(hZ, lambdai)

[U, S, V ] = svd(hZ, 'econ');
S = diag(S);
S = S - lambdai;
nnzS = sum(S > 0);

U = U(:, 1:nnzS);
V = V(:, 1:nnzS);
S = S(1:nnzS);
S = diag(S);
end

function [RMSE] = MatCompRMSERMSE(X_true, X_recoverd)
	[m, n] = size(X_true);
	RMSE = sqrt(sumsqr(X_true -  X_recoverd) / (m*n));
end

function [U, S, V] = accExactSVD_APGnc( U1, V1, U0, V0, spa, bi, k)

m = size(U1,1);
n = size(V1,1);
Afunc  = @(x) (spa*x + (1+bi)*(U1*(V1'*x)) - bi*(U0*(V0'*x)));
Atfunc = @(y) (spa'*y + (1+bi)*(V1*(U1'*y)) - bi*(V0*(U0'*y)));

rnk = min(min(m,n), ceil(1.25*k));
% rnk = k;
[U, S, V] = lansvd(Afunc, Atfunc, m, n, rnk, 'L');

U = U(:, 1:k);
V = V(:, 1:k);

S = diag(S);
S = S(1:k);
S = diag(S);

end

function [vec_M] = vec_mat(M)
vec_M = M(:);
end
