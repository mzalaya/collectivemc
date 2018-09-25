function [RSE] = MatCompRSE(X_true, X_recovered)
    RSE = norm(X_true - X_recovered, 'fro') / norm(X_true, 'fro');
end