% //
% //  exp1_300.m
% //  
% //
% //  Created by Mokhtar Zahdi Alaya on 15/02/2019.
% //
% //
clear;
clc;

%fprintf("---------------------------------------------------------------\n");
%fprintf("--------------------  Common inputs -----------\n");
%fprintf("---------------------------------------------------------------\n");

probas_collect = {[.05 .05 .05], [.1 .1 .1],    [.15 .15 .15], ...
                  [.2 .2 .2],    [.25 .25 .25], [.3 .3 .3], ...
                  [.35 .35 .35], [.4 .4 .4],    [.45 .45 .45], ...
                  [.5 .5 .5],    [.55 .55 .55], [.6 .6 .6], ...
                  [.65 .65 .65], [.7 .7 .7],    [.75 .75 .75], ...
                  [.8 .8 .8],    [.85 .85 .85], [.9 .9 .9]...
                  [.95 .95 .95], [1. 1. 1.]};

probas = [.05 .1 .15 0.2 .25 .3 .35 .4 .45 .5 ...
          .55 .6 .65 .7 .75 .8 .85 .9 .95 1.];

n_rows = 9000;
n_cols = 3000;
rank = 15;

max_iter = 1e3;
tol = 1e-6;

% decays
% decays_collect = [3.5e-1 3e-1 2.5e-1 2.5e-1];
% decays_gauss= [4.5e-1 4e-1 3.5e-1 3e-1];
% %decays_poisson = [4e-1 4e-1 4e-1 4e-1];
% decays_poisson = [3.7e-1 3.7e-1 3.75e-1 3.7e-1];
% decays_bernoulli = [4.5e-1 4e-1 3.5e-1 3e-1];
 
decays_collect = [3e-1 2.5e-1 2e-1 1.5e-1];
decays_gauss= [3.5e-1 3e-1 2.5e-1 2e-1]; 
decays_poisson = [3.5e-1 3e-1 2.5e-1 2e-1];
decays_bernoulli = [3.5e-1 3e-1 2.5e-1 2e-1];


fprintf("---------------------------------------------------------------\n'");
fprintf("--------------------  Poisson recovery-- ----------------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

RMSE_relative_poisson = [];


for cp=1:length(probas)
pause(0.5)
% datatypes
datatypes = {{'poisson', .5}};

% dimensions

nb_cols = [n_cols];

% ranks of sources
ranks = [rank];

% sparsity of the databases
probas_p = probas(cp);

% simulations
[source, mask, ~, ~, ~, ~] = CollectMCsimu(n_rows, nb_cols, ranks, probas_p, datatypes);
[m, n] = size(source);
source = source / ((max(source(:))));
% fprintf("pourcentage of observations:");% 100 * nnz(source)/(9*1e4));
% disp(100 * nnz(source)/(9*1e4));
%% Training and Testing sets

[row, col, val] = find(source);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.8));
tstIdx = idx(ceil(length(val)*0.8): end);

% training and testing sets
traData = sparse(row(traIdx), col(traIdx), val(traIdx), m, n);
tstData = sparse(row(tstIdx), col(tstIdx), val(tstIdx), m, n);

%% lambdas
[~, grad_source] = LikelihoodAndGradLikelihood(source, source, datatypes, n_cols, mask);
[~, lambda, ~] = lansvd(grad_source, 1);

%% Optimization

fprintf("---------------------------------------------------------------\n");
fprintf("------------------- PLAIS-IMPUTE for Poisson      ----------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

% para.test.row  = row(tstIdx);
% para.test.col  = col(tstIdx);
% para.test.data = val(tstIdx);
para.maxIter = max_iter;
para.tol = tol;
if cp >= 1 & cp <= 5
    para.decay = decays_poisson(1);
elseif cp >= 6 & cp <= 10
    para.decay = decays_poisson(2);
elseif cp >= 10 & cp <= 15
     para.decay = decays_poisson(3);
else
    para.decay = decays_poisson(4);
end 

para.exact = 0;
para.maxR = 5*rank;

[row_traData, col_traData] = find(traData);
mask_traData = sparse(row_traData, col_traData, ones(length(col_traData),1), m, n);

[U, S, V, output ] = PLAISImpute(traData, lambda, para, datatypes, nb_cols, mask_traData);
sln_AISoftImpute = U * S * V';

[rows_tst, cols_tst, val_tst] = find(tstData);
tst_mask = sparse(rows_tst, cols_tst, 1, m, n);
rmse = MatCompRMSE(tstData.*tst_mask, sln_AISoftImpute.*tst_mask);
RMSE_relative_poisson(cp) = rmse;
end

fprintf("---------------------------------------------------------------\n");
fprintf("--------------------  Collective recovery----------------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

RMSE_relative_collect = [];

for cp = 1:20
% datatypes
%fprintf("------------------------------PAUSE---------------------------------\n");
%disp(cp);
%pause(0.5);
datatypes = {{'gaussian', [.5, 1.]}, {'poisson', .5}, {'bernoulli', .5}};

% dimensions
nb_cols = [n_cols, n_cols, n_cols];

% ranks of sources
ranks = [rank, rank, rank];

% sparsity of the databases
probas_p = probas_collect{cp};

% simulations
[source, mask, ~, ~, ~, ~] = CollectMCsimu(n_rows, nb_cols, ranks, probas_p, datatypes);
[m, n] = size(source);
source = source / ((max(source(:))));
% fprintf("pourcentage of observations:");% 100 * nnz(source)/(9*1e4));
% disp(100 * nnz(source)/(9*1e4));

% training and testing sets
[row, col, val] = find(source);
idx = randperm(length(val));
val = val - mean(val);
val = val/std(val);
traIdx = idx(1:floor(length(val)*0.8));
tstIdx = idx(ceil(length(val)*0.8): end);
% training and testing sets
traData = sparse(row(traIdx), col(traIdx), val(traIdx), m, n);
tstData = sparse(row(tstIdx), col(tstIdx), val(tstIdx), m, n);

% lambda
[~, grad_source] = LikelihoodAndGradLikelihood(source, source, datatypes, nb_cols, mask);
[~, lambda, ~] = lansvd(grad_source, 1);

% optimization

fprintf("---------------------------------------------------------------\n");
fprintf("------------------- PLAIS-IMPUTE on collective ----------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

para.maxIter = max_iter;
para.tol = tol;
if cp >= 1 & cp <= 5
    para.decay = decays_collect(1);
elseif cp >= 6 & cp <= 10
    para.decay = decays_collect(2);
elseif cp >= 10 & cp <= 15
     para.decay = decays_collect(3);
else
    para.decay = decays_collect(4);
end
para.exact = 0;
para.maxR = 5*rank;

[row_traData, col_traData] = find(traData);
mask_traData = sparse(row_traData, col_traData, ones(length(col_traData),1), m, n);

[U, S, V, output ] = PLAISImpute(traData, lambda, para, datatypes, nb_cols, mask_traData);
sln_AISoftImpute = U * S* V';

[rows_tst, cols_tst, val_tst] = find(tstData);
tst_mask = sparse(rows_tst, cols_tst, 1, m, n);
rmse = MatCompRMSE(tstData.*tst_mask, sln_AISoftImpute.*tst_mask);
RMSE_relative_collect(cp) = rmse;
end

fprintf("---------------------------------------------------------------\n'");
fprintf("--------------------  Gaussian recovery-- ----------------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

RMSE_relative_gauss = [];

for cp=1:length(probas)
% datatypes
datatypes = {{'gaussian', [.5, 1.]}};

% dimensions
nb_cols = [n_cols];

% ranks of sources
ranks = [rank];

% sparsity of the databases
probas_p = probas(cp);

% simulations
[source, mask, ~, ~, ~, ~] = CollectMCsimu(n_rows, nb_cols, ranks, probas_p, datatypes);
[m, n] = size(source);
source = source / ((max(source(:))));
% fprintf("pourcentage of observations:");% 100 * nnz(source)/(9*1e4));
% disp(100 * nnz(source)/(9*1e4));
%% Training and Testing sets

[row, col, val] = find(source);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.8));
tstIdx = idx(ceil(length(val)*0.8): end);

% training and testing sets
traData = sparse(row(traIdx), col(traIdx), val(traIdx), m, n);
tstData = sparse(row(tstIdx), col(tstIdx), val(tstIdx), m, n);

%% lambdas
[~, grad_source] = LikelihoodAndGradLikelihood(source, source, datatypes, n_cols, mask);
[~, lambda, ~] = lansvd(grad_source, 1);

%% Optimization

fprintf("---------------------------------------------------------------\n");
fprintf("------------------- PLAIS-IMPUTE for Gaussian      ----------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

% para.test.row  = row(tstIdx);
% para.test.col  = col(tstIdx);
% para.test.data = val(tstIdx);
para.maxIter = max_iter;
para.tol = tol;
if cp >= 1 & cp <= 5
    para.decay = decays_gauss(1);
elseif cp >= 6 & cp <= 10
    para.decay = decays_gauss(2);
elseif cp >= 10 & cp <= 15
     para.decay = decays_gauss(3);
else
    para.decay = decays_gauss(4);
end 

para.exact = 0;
para.maxR = 5*rank;

[row_traData, col_traData] = find(traData);
mask_traData = sparse(row_traData, col_traData, ones(length(col_traData),1), m, n);

[U, S, V, output ] = PLAISImpute(traData, lambda, para, datatypes, nb_cols, mask_traData);
sln_AISoftImpute = U * S * V';

[rows_tst, cols_tst, val_tst] = find(tstData);
tst_mask = sparse(rows_tst, cols_tst, 1, m, n);
rmse = MatCompRMSE(tstData.*tst_mask, sln_AISoftImpute.*tst_mask);
RMSE_relative_gauss(cp) = rmse;
end

fprintf("---------------------------------------------------------------\n'");
fprintf("--------------------  Bernoulli recovery-- ----------------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

RMSE_relative_bernoulli = [];

for cp=1:length(probas)
% datatypes
datatypes = {{'bernoulli', .5}};

% dimensions

nb_cols = [n_cols];

% ranks of sources
ranks = [rank];

% sparsity of the databases
probas_p = probas(cp);

% simulations
[source, mask, ~, ~, ~, ~] = CollectMCsimu(n_rows, nb_cols, ranks, probas_p, datatypes);
[m, n] = size(source);
source = source / ((max(source(:))));
% fprintf("pourcentage of observations:");% 100 * nnz(source)/(9*1e4));
% disp(100 * nnz(source)/(9*1e4));
%% Training and Testing sets

[row, col, val] = find(source);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.8));
tstIdx = idx(ceil(length(val)*0.8): end);

% training and testing sets
traData = sparse(row(traIdx), col(traIdx), val(traIdx), m, n);
tstData = sparse(row(tstIdx), col(tstIdx), val(tstIdx), m, n);

%% lambdas
[~, grad_source] = LikelihoodAndGradLikelihood(source, source, datatypes, n_cols, mask);
[~, lambda, ~] = lansvd(grad_source, 1);

%% Optimization

fprintf("---------------------------------------------------------------\n");
fprintf("------------------- PLAIS-IMPUTE for Bernoulli     ----------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

% para.test.row  = row(tstIdx);
% para.test.col  = col(tstIdx);
% para.test.data = val(tstIdx);
para.maxIter = max_iter;
para.tol = tol;
if cp >= 1 & cp <= 5
    para.decay = decays_bernoulli(1);
elseif cp >= 6 & cp <= 10
    para.decay = decays_bernoulli(2);
elseif cp >= 10 & cp <= 15
     para.decay = decays_bernoulli(3);
else
    para.decay = decays_bernoulli(4);
end 


para.exact = 0;
para.maxR = 5*rank;

[row_traData, col_traData] = find(traData);
mask_traData = sparse(row_traData, col_traData, ones(length(col_traData),1), m, n);

[U, S, V, output ] = PLAISImpute(traData, lambda, para, datatypes, nb_cols, mask_traData);
sln_AISoftImpute = U * S * V';

[rows_tst, cols_tst, val_tst] = find(tstData);
tst_mask = sparse(rows_tst, cols_tst, 1, m, n);
rmse = MatCompRMSE(tstData.*tst_mask, sln_AISoftImpute.*tst_mask);
RMSE_relative_bernoulli(cp) = rmse;
end



RMSE_matrix_letters = [probas' RMSE_relative_collect' ...
                       RMSE_relative_gauss' RMSE_relative_poisson' ...
                       RMSE_relative_bernoulli'];
                       
                       
                       colNames = {'probas', 'RMSE_relative_collect', 'RMSE_relative_gaussian' ...
                           'RMSE_relative_poisson', 'RMSE_relative_bernoulli'};
                       RMSE_matrix_table = array2table(RMSE_matrix_letters,'VariableNames',colNames);
                       writetable(RMSE_matrix_table, 'RMSE_relatives_9000.csv','QuoteStrings',true);
                       
                       
fprintf("---------------------------------------------------------------\n");
fprintf("-------------------  PLOTS         ----------------\n");
fprintf("---------------------------------------------------------------\n\n\n");

figure
box on;
hold on;
ylim auto;
xlim([0.05 1])
plot(probas, RMSE_relative_collect , 'Marker', 'p', 'color', 'red', 'linewidth', 2, 'MarkerSize', 6);
plot(probas , RMSE_relative_gauss , 'Marker', 's', 'color', 'blue', 'linewidth', 2, 'MarkerSize', 6);
plot(probas , RMSE_relative_poisson , 'Marker', 'd', 'color', 'green', 'linewidth', 2, 'MarkerSize', 6);
plot(probas , RMSE_relative_bernoulli, 'Marker', '*', 'color', 'magenta', 'linewidth', 2, 'MarkerSize', 6);

legend({'Collective' 'Gaussian', 'Poisson', 'Bernoulli'},'FontSize',12);
xlabel('Percentage of known entries ', 'FontSize',12);
ylabel('Relative error', 'FontSize',12);
set(get(gca,'ylabel'),'rotation',90)
title('$$d_u = D = 9000$$', 'Interpreter', 'Latex', 'FontSize',12);
saveas(gcf,'table/data_9000_relative_errors.pdf');

close all;

