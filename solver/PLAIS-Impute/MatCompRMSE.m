function [RMSE] = MatCompRMSE(X_true, X_recovered)
    RMSE = norm((X_true -  X_recovered), 'fro') / (norm(X_true, 'fro'));
end
