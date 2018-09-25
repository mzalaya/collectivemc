function [RMSE] = MatCompRMSERMSE(X_true, X_recoverd)
	[m, n] = size(X_true);
	RMSE = sqrt(sumsqr(X_true -  X_recoverd) / (m*n));
end