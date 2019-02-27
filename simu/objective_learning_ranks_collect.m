% //
% //  objective_learning_ranks.m
% //
% //
% //  Created by Mokhtar Z. Alaya on 16/02/2019.
% //

clear;
clc;

% %fprintf("---------------------------------------------------------------\n");
% %fprintf("--------------------  Common inputs -----------\n");
% %fprintf("---------------------------------------------------------------\n");
% 
% p = .2;
% max_iter = 1e3;
% tol = 1e-7;
% decay_collect1 = 3e-1;
% 
% % fprintf("---------------------------------------------------------------\n");
% fprintf("--------------------  Experience 1-----------------------------\n");
% fprintf("---------------------------------------------------------------\n\n");
% 
% n_rows1 = 3000;
% n_cols1= 1000;
% rank1 = 5;
% 
% % datatypes
% % 
% datatypes = {{'gaussian', [.5, 1.]}, {'poisson', .5}, {'bernoulli', .5}};
% % 
% % dimensions
% nb_cols1 = [n_cols1, n_cols1, n_cols1];
% 
% % ranks of sources
% ranks1 = [rank1, rank1, rank1];
% 
% % sparsity of the databases
% probas_p = [p, p, p];
% 
% % simulations
% [source1, mask1, ~, ~, ~, ~] = CollectMCsimu(n_rows1, nb_cols1, ranks1, probas_p, datatypes);
% [m1, n1] = size(source1);
% source1 = source1 / ((max(source1(:))));
% 
% 
% % training and testing sets
% [row1, col1, val1] = find(source1);
% idx1 = randperm(length(val1));
% val1 = val1 - mean(val1);
% val1 = val1/std(val1);
% traIdx1 = idx1(1:floor(length(val1)*0.8));
% tstIdx1 = idx1(ceil(length(val1)*0.8): end);
% % training and testing sets
% traData1 = sparse(row1(traIdx1), col1(traIdx1), val1(traIdx1), m1, n1);
% tstData1 = sparse(row1(tstIdx1), col1(tstIdx1), val1(tstIdx1), m1, n1);
% 
% % lambda
% [~, grad_source1] = LikelihoodAndGradLikelihood(source1, source1, datatypes, nb_cols1, mask1);
% [~, lambda1, ~] = lansvd(grad_source1, 1);
% 
% % optimization
% 
% para1.maxIter = max_iter;
% para1.tol = tol;
% para1.decay = decay_collect1
% para1.exact = 0;
% para1.maxR = 5*rank1;
% 
% [row_traData1, col_traData1] = find(traData1);
% mask_traData1 = sparse(row_traData1, col_traData1, ones(length(col_traData1),1), m1, n1);
% 
% [U1, S1, V1, output1 ] = PLAISImpute(traData1, lambda1, para1, datatypes, nb_cols1, mask_traData1);
% sln_AISoftImpute1= U1* S1* V1';
% 
% obj1 = output1.obj;
% ranksIn1 = output1.RankIn;
% ranksOut1 = output1.RankOut;
% times1 = output1.Time;
% lambdas1 = output1.lambdas;
% 
% outputs_3000 = [obj1 ranksIn1 ranksOut1 times1 lambdas1]; 
% colNames1 = {'Objective_3000', 'RankIn_3000', 'RankOut_3000', 'Times_3000', 'Lambdas_3000'};
% outputs_3000 = array2table(outputs_3000,'VariableNames',colNames1);
% writetable(outputs_3000, 'table/Outputs_table_3000.csv','QuoteStrings',true);
% 
% fprintf("\n\n ---------------------------------------------------------------\n");
% fprintf("--------------------  Experience 2 ---------------------------------\n");
% fprintf("---------------------------------------------------------------\n\n");
% 
% n_rows2 = 6000;
% n_cols2= 2000;
% rank2 = 10;
% decay_collect2 = 3e-1;
% % dimensions
% nb_cols2 = [n_cols2, n_cols2, n_cols2];
% 
% % ranks of sources
% ranks2 = [rank2, rank2, rank2];
% 
% % sparsity of the databases
% probas_p = [p, p, p];
% 
% % simulations
% [source2, mask2, ~, ~, ~, ~] = CollectMCsimu(n_rows2, nb_cols2, ranks2, probas_p, datatypes);
% [m2, n2] = size(source2);
% source2 = source2 / ((max(source2(:))));
% 
% 
% % training and testing sets
% [row2, col2, val2] = find(source2);
% idx2 = randperm(length(val2));
% val2 = val2 - mean(val2);
% val2 = val2/std(val2);
% traIdx2 = idx2(1:floor(length(val2)*0.8));
% tstIdx2 = idx2(ceil(length(val2)*0.8): end);
% % training and testing sets
% traData2 = sparse(row2(traIdx2), col2(traIdx2), val2(traIdx2), m2, n2);
% tstData2 = sparse(row2(tstIdx2), col2(tstIdx2), val2(tstIdx2), m2, n2);
% 
% % lambda
% [~, grad_source2] = LikelihoodAndGradLikelihood(source2, source2, datatypes, nb_cols2, mask2);
% [~, lambda2, ~] = lansvd(grad_source2, 1);
% 
% % optimization
% 
% para2.maxIter = max_iter;
% para2.tol = tol;
% para2.decay = decay_collect2;
% para2.exact = 0;
% para2.maxR = 5*rank2;
% 
% [row_traData2, col_traData2] = find(traData2);
% mask_traData2 = sparse(row_traData2, col_traData2, ones(length(col_traData2),1), m2, n2);
% 
% [U2, S2, V2, output2 ] = PLAISImpute(traData2, lambda2, para2, datatypes, nb_cols2, mask_traData2);
% sln_AISoftImpute2= U2* S2* V2';
% 
% obj2 = output2.obj;
% ranksIn2 = output2.RankIn;
% ranksOut2 = output2.RankOut;
% times2 = output2.Time;
% lambdas2 = output2.lambdas;
% 
% outputs_6000 = [obj2 ranksIn2 ranksOut2 times2 lambdas2]; 
% 
% colNames2 = {'Objective_6000', 'RankIn_6000', 'RankOut_6000', 'Times_6000', 'Lambdas_6000'};
% outputs_6000 = array2table(outputs_6000,'VariableNames',colNames2);
% writetable(outputs_6000, 'table/Outputs_table_6000.csv','QuoteStrings',true);

fprintf("\n\n---------------------------------------------------------------\n");
fprintf("--------------------  Experience 3 ---------------------------------\n");
fprintf("---------------------------------------------------------------\n\n\n");
tol = 1e-7;
max_iter = 80;
p = .6;
datatypes = {{'gaussian', [.5, 1.]}, {'poisson', .5}, {'bernoulli', .5}};
n_rows3 = 9000;
n_cols3= 3000;
rank3 = 15;
decay_collect3 = 2.5e-1;
% dimensions
nb_cols3 = [n_cols3, n_cols3, n_cols3];

% ranks of sources
ranks3 = [rank3, rank3, rank3];

% sparsity of the databases
probas_p = [p, p, p];

% simulations
[source3, mask3, ~, ~, ~, ~] = CollectMCsimu(n_rows3, nb_cols3, ranks3, probas_p, datatypes);
[m3, n3] = size(source3);
source3 = source3 / ((max(source3(:))));


% training and testing sets
[row3, col3, val3] = find(source3);
idx3 = randperm(length(val3));
val3 = val3 - mean(val3);
val3 = val3/std(val3);
traIdx3 = idx3(1:floor(length(val3)*0.8));
tstIdx3 = idx3(ceil(length(val3)*0.8): end);
% training and testing sets
traData3 = sparse(row3(traIdx3), col3(traIdx3), val3(traIdx3), m3, n3);
tstData3 = sparse(row3(tstIdx3), col3(tstIdx3), val3(tstIdx3), m3, n3);

% lambda
[~, grad_source3] = LikelihoodAndGradLikelihood(source3, source3, datatypes, nb_cols3, mask3);
[~, lambda3, ~] = lansvd(grad_source3, 1);

% optimization

para3.maxIter = 1e3;
para3.tol = tol;
para3.decay = decay_collect3;
para3.maxR = 5*rank3;

[row_traData3, col_traData3] = find(traData3);
mask_traData3 = sparse(row_traData3, col_traData3, ones(length(col_traData3),1), m3, n3);

[U3, S3, V3, output3 ] = PLAISImpute(traData3, lambda3, para3, datatypes, nb_cols3, mask_traData3);
sln_AISoftImpute3= U3* S3* V3';

obj3 = output3.obj;
ranksIn3 = output3.RankIn;
ranksOut3 = output3.RankOut;
times3 = output3.Time;
lambdas3 = output3.lambdas;

outputs_9000 = [obj3 ranksIn3 ranksOut3 times3 lambdas3]; 

colNames3 = {'Objective_9000', 'RankIn_9000', 'RankOut_9000', 'Times_9000', 'Lambdas_9000'};
outputs_9000 = array2table(outputs_9000,'VariableNames',colNames3);
writetable(outputs_9000, 'table/Outputs_table_9000.csv','QuoteStrings',true);
