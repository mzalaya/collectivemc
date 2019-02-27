% //
% //  exp1_3000_coldStart.m
% //
% //
% //

clear;
clc;

% fprintf("---------------------------------------------------------------\n");
% fprintf("--------------------  Common inputs -----------\n");
% fprintf("---------------------------------------------------------------\n");
p = .6;
n_rows = 3000;
n_cols = 1000;
rank = 5;
max_iter = 200;
tol = 1e-6;
mcMax = 10; 
decay_collect = 3e-1;
decay_gaussian = 3.5e-1;
decay_poisson = 3.5e-1;
decay_bernoulli = 3.5e-1;

RMSE_relative_collect_1 = [];
RMSE_relative_collect_2 = [];
RMSE_relative_collect_3 = [];
RMSE_relative_gaussian = [];
RMSE_relative_poisson = [];
RMSE_relative_bernoulli = [];

data_sparsity_collect1 = [];
data_sparsity_gaussian = [];
data_sparsity_collect2 = [];
data_sparsity_poisson = [];
data_sparsity_collect3 = [];
data_sparsity_bernoulli = [];

for cp =1:mcMax
    %% datatypes
    datatypes = {{'gaussian', [.5,  1.]}, {'poisson', .5}, {'bernoulli', 0.5}};
    %% dimensions

    nb_cols = [n_cols, n_cols, n_cols];

    %% ranks
    ranks = [rank, rank, rank];

    %% probas
    probas = [p, p, p];

    %% simulations
    [source1, mask1, ~, ~, ~, ~] = ...
    CollectMCsimu(n_rows, nb_cols, ranks, probas, datatypes);
    source1 = source1 / max(source1(:));
    [m, n] = size(source1);
    
    source_gaussian1 = source1(:,1:1000);
    source_poisson1 = source1(:,1001:2000);
    source_bernoulli1 = source1(:,2001:3000);

    % cold-start for Poisson data
    [row_gaussian, col_gaussian, data_source_gaussian] = find(source_gaussian1);
    
    % fprintf("---------------------------------------------------------------\n");
    % fprintf("-------------------  SPARSITY  -- --------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");
    
    data_source_gaussian(1:floor(size(data_source_gaussian)/5),:) = 0.;
    [m_gauss n_gauss] = size(source_gaussian1);
    source_gaussian = sparse(row_gaussian, col_gaussian, data_source_gaussian, m_gauss, n_gauss);
    
    data_sparsity_for_gaussian = nnz(source_gaussian)/(3*1e6);
    % fprintf("data sparsity for Poisson data is %d\n", data_sparsity_for_gaussian);
    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- Collective cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    M_collective1 = horzcat(source_gaussian, source_poisson1);
    source1 = horzcat(M_collective1, source_bernoulli1);
    data_sparsity_for_collective_1 = nnz(source1)/(9*1e6);

    data_sparsity_collect1(cp) = data_sparsity_for_collective_1;
    data_sparsity_gaussian(cp) = data_sparsity_for_gaussian;

    % fprintf("data sparsity for collective data is %d\n", data_sparsity_for_collective_1);
    % append(data_sparsity1, data_sparsity_for_collective_1, data_sparsity_for_gaussian);

    % training and testing sets
    [row1, col1, val1] = find(source1);
    idx1 = randperm(length(val1));
    val1 = val1 - mean(val1);
    val1 = val1/std(val1);
    traIdx1 = idx1(1:floor(length(val1)*0.8));
    tstIdx1 = idx1(ceil(length(val1)*0.8): end);
    % training and testing sets
    traData1 = sparse(row1(traIdx1), col1(traIdx1), val1(traIdx1), m, n);
    tstData1 = sparse(row1(tstIdx1), col1(tstIdx1), val1(tstIdx1), m, n);

    % lambda
    [~, grad_source1] = LikelihoodAndGradLikelihood(source1, source1, datatypes, nb_cols, mask1);
    [~, lambda1, ~] = lansvd(grad_source1, 1);

    % optimization

    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- PLAIS-IMPUTE on collective cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    para1.maxIter = max_iter;
    para1.tol = tol;
    para1.decay = decay_collect;
    para1.exact = 0;
    para1.maxR = 5*rank;

    [row_traData1, col_traData1] = find(traData1);
    mask_traData1 = sparse(row_traData1, col_traData1, 1, m, n);

    [U1, S1, V1, output1 ] = PLAISImpute(traData1, lambda1, para1, datatypes, nb_cols, mask_traData1);
    sln_AISoftImpute1 = U1 * S1* V1';

    [rows_tst1, cols_tst1, val_tst1] = find(tstData1);
    tst_mask1 = sparse(rows_tst1, cols_tst1, 1, m, n);
    rmse_collect1 = MatCompRMSE(tstData1.*tst_mask1, sln_AISoftImpute1.*tst_mask1);
    RMSE_relative_collect1(cp) = rmse_collect1;

   
    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- Gaussian cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    %% training and testing sets
    val_gaussian = data_source_gaussian;
    idx_gaussian = randperm(length(val_gaussian));

    val_gaussian= val_gaussian - mean(val_gaussian);
    val_gaussian = val_gaussian/std(val_gaussian);

    traIdx_gaussian= idx_gaussian(1:floor(length(val_gaussian)*0.8));
    tstIdx_gaussian = idx_gaussian(ceil(length(val_gaussian)*0.8): end);

    % training and testing sets
    traData_gaussian = sparse(row_gaussian(traIdx_gaussian), col_gaussian(traIdx_gaussian), val_gaussian(traIdx_gaussian), n_rows, n_cols);
    tstData_gaussian= sparse(row_gaussian(tstIdx_gaussian), col_gaussian(tstIdx_gaussian), val_gaussian(tstIdx_gaussian), n_rows, n_cols);
    
    [row_traData_gaussian, col_traData_gaussian, ~] = find(traData_gaussian);
    mask_traData_gaussian = sparse(row_traData_gaussian, col_traData_gaussian, 1, n_rows, n_cols);


    %% lambdas
    datatypes_gaussian = {{'gaussian', [.5,  1.]}};
    n_cols_gaussian = [n_cols];
    [~, grad_source_gaussian] = LikelihoodAndGradLikelihood(source_gaussian, source_gaussian, datatypes_gaussian, n_cols_gaussian, mask_traData_gaussian);
    [~, lambda_gaussian, ~] = lansvd(grad_source_gaussian, 1);

     % optimization

    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- PLAIS-IMPUTE on Gaussian cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    para_gaussian.maxIter = max_iter;
    para_gaussian.tol = tol;
    para_gaussian.decay = decay_gaussian;
    para_gaussian.exact = 0;
    para_gaussian.maxR = 5*rank;
        
    [U_gaussian, S_gaussian, V_gaussian, output_gaussian ] = PLAISImpute(traData_gaussian, lambda_gaussian, ...
                                                                      para_gaussian, datatypes_gaussian, n_cols_gaussian, mask_traData_gaussian );
    sln_AISoftImpute_gaussian = U_gaussian *  S_gaussian * V_gaussian';

    [rows_tst_gaussian, cols_tst_gaussian, ~] = find(tstData_gaussian);
    tst_mask_gaussian = sparse(rows_tst_gaussian, cols_tst_gaussian, 1, n_rows, n_cols);
    
    rmse_gaussian = MatCompRMSE(tstData_gaussian.*tst_mask_gaussian, sln_AISoftImpute_gaussian.*tst_mask_gaussian);
    RMSE_relative_gaussian(cp) = rmse_gaussian;
    
end


%-------------------------------------------------------------------------------
for cp =1:mcMax
    %% datatypes
    datatypes = {{'gaussian', [.5,  1.]}, {'poisson', .5}, {'bernoulli', 0.5}};
    %% dimensions

    nb_cols = [n_cols, n_cols, n_cols];

    %% ranks
    ranks = [rank, rank, rank];

    %% probas
    probas = [p, p, p];

    %% simulations
    [source2, mask2, ~, ~, ~, ~] = ...
    CollectMCsimu(n_rows, nb_cols, ranks, probas, datatypes);
    source2 = source2 / max(source2(:));
    [m, n] = size(source2);
    
    source_gaussian2 = source2(:,1:1000);
    source_poisson2 = source2(:,1001:2000);
    source_bernoulli2 = source2(:,2001:3000);

    % cold-start for Poisson data
    [row_poisson, col_poisson, data_source_poisson] = find(source_poisson2);

    % fprintf("---------------------------------------------------------------\n");
    % fprintf("-------------------  SPARSITY  -- --------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");
    
    data_source_poisson(2:floor(size(data_source_poisson)/5),:) = 0.;
    [m_pois n_pois] = size(source_poisson2);
    source_poisson = sparse(row_poisson, col_poisson, data_source_poisson, m_pois, n_pois);
    
    data_sparsity_for_poisson = nnz(source_poisson)/(3*1e6);
    % fprintf("data sparsity for Poisson data is %d\n", data_sparsity_for_poisson);

    
    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- Collective cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    M_collective2 = horzcat(source_gaussian2, source_poisson);
    source2 = horzcat(M_collective2, source_bernoulli2);

    data_sparsity_for_collective_2 = nnz(source2)/(9*1e6);

    data_sparsity_collect2(cp) = data_sparsity_for_collective_2;
    data_sparsity_poisson(cp) = data_sparsity_for_poisson;

    % fprintf("data sparsity for collective data is %d\n", data_sparsity_for_collective_2);

    % training and testing sets
    [row2, col2, val2] = find(source2);
    idx2 = randperm(length(val2));
    val2 = val2 - mean(val2);
    val2 = val2/std(val2);
    traIdx2 = idx2(1:floor(length(val2)*0.8));
    tstIdx2 = idx2(ceil(length(val2)*0.8): end);
    % training and testing sets
    traData2 = sparse(row2(traIdx2), col2(traIdx2), val2(traIdx2), m, n);
    tstData2 = sparse(row2(tstIdx2), col2(tstIdx2), val2(tstIdx2), m, n);

    % lambda
    [~, grad_source2] = LikelihoodAndGradLikelihood(source2, source2, datatypes, nb_cols, mask2);
    [~, lambda2, ~] = lansvd(grad_source2, 1);

    % optimization

    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- PLAIS-IMPUTE on collective cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    para2.maxIter = max_iter;
    para2.tol = tol;
    para2.decay = decay_collect;
    para2.exact = 0;
    para2.maxR = 5*rank;

    [row_traData2, col_traData2] = find(traData2);
    mask_traData2 = sparse(row_traData2, col_traData2, 1, m, n);

    [U2, S2, V2, output2 ] = PLAISImpute(traData2, lambda2, para2, datatypes, nb_cols, mask_traData2);
    sln_AISoftImpute2 = U2 * S2* V2';

    [rows_tst2, cols_tst2, val_tst2] = find(tstData2);
    tst_mask2 = sparse(rows_tst2, cols_tst2, 1, m, n);
    rmse_collect2 = MatCompRMSE(tstData2.*tst_mask2, sln_AISoftImpute2.*tst_mask2);
    RMSE_relative_collect2(cp) = rmse_collect2;

   
    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- Poisson cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    %% training and testing sets
    val_poisson = data_source_poisson;
    idx_poisson = randperm(length(val_poisson));

    val_poisson= val_poisson - mean(val_poisson);
    val_poisson = val_poisson/std(val_poisson);

    traIdx_poisson= idx_poisson(1:floor(length(val_poisson)*0.8));
    tstIdx_poisson = idx_poisson(ceil(length(val_poisson)*0.8): end);

    % training and testing sets
    traData_poisson = sparse(row_poisson(traIdx_poisson), col_poisson(traIdx_poisson), val_poisson(traIdx_poisson), n_rows, n_cols);
    tstData_poisson= sparse(row_poisson(tstIdx_poisson), col_poisson(tstIdx_poisson), val_poisson(tstIdx_poisson), n_rows, n_cols);
    
    [row_traData_poisson, col_traData_poisson, ~] = find(traData_poisson);
    mask_traData_poisson = sparse(row_traData_poisson, col_traData_poisson, 1, n_rows, n_cols);


    %% lambdas
    datatypes_poisson = {{'poisson',.5}};
    n_cols_poisson = [n_cols];
    [~, grad_source_poisson] = LikelihoodAndGradLikelihood(source_poisson, source_poisson, datatypes_poisson, n_cols_poisson, mask_traData_poisson);
    [~, lambda_poisson, ~] = lansvd(grad_source_poisson, 1);

     % optimization

    % fprintf("---------------------------------------------------------------\n");
    % fprintf("------------------- PLAIS-IMPUTE on poisson cold ----------------\n");
    % fprintf("---------------------------------------------------------------\n\n\n");

    para_poisson.maxIter = max_iter;
    para_poisson.tol = tol;
    para_poisson.decay = decay_poisson;
    para_poisson.exact = 0;
    para_poisson.maxR = 5*rank;
        
    [U_poisson, S_poisson, V_poisson, output_poisson ] = PLAISImpute(traData_poisson, lambda_poisson, ...
                                                                      para_poisson, datatypes_poisson, n_cols_poisson, mask_traData_poisson );
    sln_AISoftImpute_poisson = U_poisson *  S_poisson * V_poisson';

    [rows_tst_poisson, cols_tst_poisson, ~] = find(tstData_poisson);
    tst_mask_poisson = sparse(rows_tst_poisson, cols_tst_poisson, 1, n_rows, n_cols);
    
    rmse_poisson = MatCompRMSE(tstData_poisson.*tst_mask_poisson, sln_AISoftImpute_poisson.*tst_mask_poisson);
    RMSE_relative_poisson(cp) = rmse_poisson;
    
end

%-------------------------------------------------------------------------------
for cp =1:mcMax
%% datatypes
datatypes = {{'gaussian', [.5,  1.]}, {'poisson', .5}, {'bernoulli', 0.5}};
%% dimensions

nb_cols = [n_cols, n_cols, n_cols];

%% ranks
ranks = [rank, rank, rank];

%% probas
probas = [p, p, p];

%% simulations
[source3, mask3, ~, ~, ~, ~] = ...
CollectMCsimu(n_rows, nb_cols, ranks, probas, datatypes);
source3 = source3 / max(source3(:));
[m, n] = size(source3);

source_gaussian3 = source3(:,1:1000);
source_poisson3 = source3(:,1001:2000);
source_bernoulli3 = source3(:,2001:3000);

% cold-start for bernoulli data
[row_bernoulli, col_bernoulli, data_source_bernoulli] = find(source_bernoulli3);

% fprintf("---------------------------------------------------------------\n");
% fprintf("-------------------  SPARSITY  -- --------------\n");
% fprintf("---------------------------------------------------------------\n\n\n");

data_source_bernoulli(1:floor(size(data_source_bernoulli)/5),:) = 0.;
[m_pois n_pois] = size(source_bernoulli3);
source_bernoulli = sparse(row_bernoulli, col_bernoulli, data_source_bernoulli, m_pois, n_pois);

data_sparsity_for_bernoulli = nnz(source_bernoulli)/(3*1e6);
fprintf("data sparsity for bernoulli data is %d\n", data_sparsity_for_bernoulli);


% fprintf("---------------------------------------------------------------\n");
% fprintf("------------------- Collective cold ----------------\n");
% fprintf("---------------------------------------------------------------\n\n\n");

M_collective3 = horzcat(source_gaussian3, source_poisson3);
source3 = horzcat(M_collective3, source_bernoulli3);
data_sparsity_for_collective_3 = nnz(source3)/(9*1e6);

data_sparsity_collect3(cp) = data_sparsity_for_collective_3;
data_sparsity_bernoulli(cp) = data_sparsity_for_bernoulli;
% fprintf("data sparsity for collective data is %d\n", data_sparsity_for_collective_3);

% training and testing sets
[row3, col3, val3] = find(source3);
idx3 = randperm(length(val3));
val3 = val3 - mean(val3);
val3 = val3/std(val3);
traIdx3 = idx3(1:floor(length(val3)*0.8));
tstIdx3 = idx3(ceil(length(val3)*0.8): end);
% training and testing sets
traData3 = sparse(row3(traIdx3), col3(traIdx3), val3(traIdx3), m, n);
tstData3 = sparse(row3(tstIdx3), col3(tstIdx3), val3(tstIdx3), m, n);

% lambda
[~, grad_source3] = LikelihoodAndGradLikelihood(source3, source3, datatypes, nb_cols, mask3);
[~, lambda3, ~] = lansvd(grad_source3, 1);

% optimization

% fprintf("---------------------------------------------------------------\n");
% fprintf("------------------- PLAIS-IMPUTE on collective cold ----------------\n");
% fprintf("---------------------------------------------------------------\n\n\n");

para3.maxIter = max_iter;
para3.tol = tol;
para3.decay = decay_collect;
para3.exact = 0;
para3.maxR = 5*rank;

[row_traData3, col_traData3] = find(traData3);
mask_traData3 = sparse(row_traData3, col_traData3, 1, m, n);

[U3, S3, V3, output3 ] = PLAISImpute(traData3, lambda3, para3, datatypes, nb_cols, mask_traData3);
sln_AISoftImpute3 = U3 * S3* V3';

[rows_tst3, cols_tst3, val_tst3] = find(tstData3);
tst_mask3 = sparse(rows_tst3, cols_tst3, 1, m, n);
rmse_collect3 = MatCompRMSE(tstData3.*tst_mask3, sln_AISoftImpute3.*tst_mask3);
RMSE_relative_collect3(cp) = rmse_collect3;


% fprintf("---------------------------------------------------------------\n");
% fprintf("------------------- Bernoulli cold ----------------\n");
% fprintf("---------------------------------------------------------------\n\n\n");

%% training and testing sets
val_bernoulli = data_source_bernoulli;
idx_bernoulli = randperm(length(val_bernoulli));

val_bernoulli= val_bernoulli - mean(val_bernoulli);
val_bernoulli = val_bernoulli/std(val_bernoulli);

traIdx_bernoulli= idx_bernoulli(1:floor(length(val_bernoulli)*0.8));
tstIdx_bernoulli = idx_bernoulli(ceil(length(val_bernoulli)*0.8): end);

% training and testing sets
traData_bernoulli = sparse(row_bernoulli(traIdx_bernoulli), col_bernoulli(traIdx_bernoulli), val_bernoulli(traIdx_bernoulli), n_rows, n_cols);
tstData_bernoulli= sparse(row_bernoulli(tstIdx_bernoulli), col_bernoulli(tstIdx_bernoulli), val_bernoulli(tstIdx_bernoulli), n_rows, n_cols);

[row_traData_bernoulli, col_traData_bernoulli, ~] = find(traData_bernoulli);
mask_traData_bernoulli = sparse(row_traData_bernoulli, col_traData_bernoulli, 1, n_rows, n_cols);


%% lambdas
datatypes_bernoulli = {{'bernoulli',.5}};
n_cols_bernoulli = [n_cols];
[~, grad_source_bernoulli] = LikelihoodAndGradLikelihood(source_bernoulli, source_bernoulli, datatypes_bernoulli, n_cols_bernoulli, mask_traData_bernoulli);
[~, lambda_bernoulli, ~] = lansvd(grad_source_bernoulli, 1);

% optimization

% fprintf("---------------------------------------------------------------\n");
% fprintf("------------------- PLAIS-IMPUTE on bernoulli cold ----------------\n");
% fprintf("---------------------------------------------------------------\n\n\n");

para_bernoulli.maxIter = max_iter;
para_bernoulli.tol = tol;
para_bernoulli.decay = decay_bernoulli;
para_bernoulli.exact = 0;
para_bernoulli.maxR = 5*rank;

[U_bernoulli, S_bernoulli, V_bernoulli, output_bernoulli ] = PLAISImpute(traData_bernoulli, lambda_bernoulli, ...
                                                                 para_bernoulli, datatypes_bernoulli, n_cols_bernoulli, mask_traData_bernoulli );
sln_AISoftImpute_bernoulli = U_bernoulli *  S_bernoulli * V_bernoulli';

[rows_tst_bernoulli, cols_tst_bernoulli, ~] = find(tstData_bernoulli);
tst_mask_bernoulli = sparse(rows_tst_bernoulli, cols_tst_bernoulli, 1, n_rows, n_cols);

rmse_bernoulli = MatCompRMSE(tstData_bernoulli.*tst_mask_bernoulli, sln_AISoftImpute_bernoulli.*tst_mask_bernoulli);
RMSE_relative_bernoulli(cp) = rmse_bernoulli;

end


fprintf("\n\n--------------------------------------------------------------------------\n");
fprintf("---------------------------- RMSES COLD MATRIX 1----------------------------\n");
fprintf("--------------------------------------------------------------------------\n\n");

disp(RMSE_relative_collect1);
%disp(mean(RMSE_relative_collect1));
%disp(std(RMSE_relative_collect1));
%disp(RMSE_relative_collect1)

fprintf("\n\n--------------------------------------------------------------------------\n");
fprintf("---------------------------- RMSES Gaussian ----------------------------\n");
fprintf("--------------------------------------------------------------------------\n\n");

disp(RMSE_relative_gaussian);
%disp(mean(RMSE_relative_gaussian));
%disp(std(RMSE_relative_gaussian));

disp(size(RMSE_relative_gaussian));
fprintf("\n\n--------------------------------------------------------------------------\n");
fprintf("---------------------------- RMSES COLD MATRIX 2----------------------------\n");
fprintf("--------------------------------------------------------------------------\n\n");

disp(RMSE_relative_collect2);
%disp(mean(RMSE_relative_collect2));
%disp(std(RMSE_relative_collect2));

fprintf("\n\n--------------------------------------------------------------------------\n");
fprintf("---------------------------- RMSES Poisson ----------------------------\n");
fprintf("--------------------------------------------------------------------------\n\n");

disp(RMSE_relative_poisson);
%disp(mean(RMSE_relative_poisson));
%disp(std(RMSE_relative_poisson));

fprintf("\n\n--------------------------------------------------------------------------\n");
fprintf("---------------------------- RMSES COLD MATRIX 3----------------------------\n");
fprintf("--------------------------------------------------------------------------\n\n");

disp(RMSE_relative_collect3);
%disp(mean(RMSE_relative_collect3));
%disp(std(RMSE_relative_collect3));

fprintf("\n\n--------------------------------------------------------------------------\n");
fprintf("---------------------------- RMSES Bernoulli ----------------------------\n");
fprintf("--------------------------------------------------------------------------\n\n");

disp(RMSE_relative_bernoulli);
%disp(mean(RMSE_relative_bernoulli));
%disp(std(RMSE_relative_bernoulli));


disp(size(RMSE_relative_collect1));
disp(size(RMSE_relative_collect2));
disp(size(RMSE_relative_collect3));
disp(size(RMSE_relative_gaussian));
disp(size(RMSE_relative_poisson));
disp(size(RMSE_relative_bernoulli));

disp(size(data_sparsity1'));
disp(size(data_sparsity2'));
disp(size(data_sparsity3'));

Data_sparsity = [data_sparsity_collect1' data_sparsity_gaussian' ...
                 data_sparsity_collect2' data_sparsity_poisson' ...
                 data_sparsity_collect3' data_sparsity_bernoulli'];
                 
colnames_sparsity = {'Sparsity_M_collect1', 'Sparsity_M_gaussian', ...
                     'Sparsity_M_collect2'  'Sparsity_M_poisson', ...
                     'Sparsity_M_collect3'  'Sparsity_M_bernoulli'};
data_spartsity_table = array2table(Data_sparsity, 'VariableNames', colnames_sparsity);
writetable(data_spartsity_table, 'table/Data_Sparsity_table_3000.csv','QuoteStrings',true);

RMSE_matrix = [RMSE_relative_collect1' RMSE_relative_gaussian' ...
               RMSE_relative_collect2' RMSE_relative_poisson' ...
               RMSE_relative_collect3' RMSE_relative_bernoulli'];


colNames = {'RMSE_relative_collect1', 'RMSE_relative_gaussian' ...
             'RMSE_relative_collect2', 'RMSE_relative_poisson' ...
             'RMSE_relative_collect3', 'RMSE_relative_bernoulli'};
RMSE_matrix_table = array2table(RMSE_matrix,'VariableNames',colNames);
writetable(RMSE_matrix_table, 'table/RMSE_coldStart_table_3000.csv','QuoteStrings',true);
