clear;
clc;
m = 1; 
obj = {};
RanksIn = {};
RanksOut = {};
RMSE = {};
Time = {};
lambdas = {};
RMSE_montecarlo = zeros(5,1);

obj_poisson = {};
RanksIn_poisson = {};
RanksOut_poisson = {};
RMSE_poisson = {};
Time_poisson = {};
lambdas_poisson = {};
RMSE_montecarlo_poisson = zeros(5,1);


while m <= 5
    fprintf("Number of iteration in the montecarlo %d:\n", m);
    %% datatypes
 	datatypes = {{'gaussian', [1.,  .05]}, {'poisson', 1.}, {'bernoulli', 0.5}};
    %% dimensions
	n_rows = 3000;
	n_cols = [1000, 1000, 1000];
    
    %% ranks
	rank = 5;
	ranks = [rank, rank, rank];
   
    %% probas
	probas = [.05, .05, .05];   
    %% simulations
	[source, ~, ~, ~, ~, ~] = ...
	CollectMCsimu(n_rows, n_cols, ranks, probas, datatypes);
    source = source / max(source(:));
    source_gaussian = source(:,1:1000);
    source_poisson = source(:,1001:2000);
    source_bernoulli = source(:,2001:3000);
    
    %% cold-start for Poisson data
    [row_poisson, col_poisson, data_source_poisson] = find(source_poisson);
    data_source_poisson(1:1000,:) = 0.0000;
    %disp(row_poisson(50000)
    %error('stop');
    source_poisson = sparse(row_poisson, col_poisson, data_source_poisson, 3000, 1000);
    data_sparsity_for_poisson = nnz(source_poisson)/(3*1e6);
    fprintf("data sparsity for Poisson data is %d\n", data_sparsity_for_poisson);
    
    %% collective-cold
     M_collective = horzcat(source_gaussian, source_poisson);
     source = horzcat(M_collective, source_bernoulli);
     data_sparsity_for_collective = nnz(source)/(9*1e6);
     fprintf("data sparsity for collective data is %d\n", data_sparsity_for_collective);
     
    %% training and testing sets
	[row, col, val] = find(source);
	idx = randperm(length(val));
    %
	% 
    val = val - mean(val);
	% 
    val = val/std(val);
    %
	traIdx = idx(1:floor(length(val)*0.5));
	tstIdx = idx(ceil(length(val)*0.5): end);

    % training and testing sets
	traData = sparse(row(traIdx), col(traIdx), val(traIdx));
	tstData = sparse(row(tstIdx), col(tstIdx), val(tstIdx));

    %% lambdas
    [~, grad_source] = LikelihoodAndGradLikelihood(source, source, datatypes, n_cols);
    [~, lambda, ~] = lansvd(grad_source, 1);
   
    %% Optimization
	disp('%------------------------------------------------------------%');
	disp('%-                        PLAIS-Impute                      -%');
	disp('%------------------------------------------------------------%');
	 
%     para.test.row  = row(tstIdx);
% 	para.test.col  = col(tstIdx);
% 	para.test.data = val(tstIdx);
    para.test.test = false;
	para.maxIter = 50;
	para.tol = 1e-6;
    
	para.decay = 3.54e-1; % 3.4498e-1; % 3.123e-1;%3.445e-1;%3.5e-1; %2.5e-1;% 319e-1; %4e-1; %3.8e-1; %3.6e-1%4e-1;
    % the ideal is 3.5e-1 for the dataset whole 3.48% 3.44
    % 3.45e-1; %  gaussian
	para.exact = 0;
	para.maxR = 5*rank;
    
	% 
     [U, S, V, output ] = AccSoftImputeFastProx(traData, lambda, para, datatypes, n_cols );
	 sln_AISoftImpute = U *  S * V';
     obj{end+1} = output.obj;
     RanksIn{end+1} = output.RankIn;
     RanksOut{end+1} = output.RankOut;
     RMSE{end+1} = output.RMSE;
     Time{end+1} = output.Time;
     lambdas{end+1} = output.lambdas;
     recovered_data = zeros(length(row(tstIdx)),1);
     rows = [row(tstIdx)];
     cols = [col(tstIdx)];
     for k = 1:length(rows)
         recovered_data(k) = sln_AISoftImpute(rows(k), cols(k));
     end
     sln_AISoftImpute_tst = sparse(rows, cols, recovered_data, 3000, 3000);
     RMSE_montecarlo(m) = MatCompRMSERMSE(tstData, sln_AISoftImpute_tst);
    
 
	% m = m +1;
    % RMSE_montecarlo(m) = MatCompRMSERMSE(tstData, sln_AISoftImpute);
    
     %% training and testing sets
	% [row_poisson, col_poisson, val_poisson] = find(source_poisson);
    val_poisson = data_source_poisson;
	idx_poisson = randperm(length(val_poisson));
    %
	% 
    val_poisson = val_poisson - mean(val_poisson);
	% 
    val_poisson = val_poisson/std(val_poisson);
    %
	traIdx_poisson = idx_poisson(1:floor(length(val_poisson)*0.5));
	tstIdx_poisson = idx_poisson(ceil(length(val_poisson)*0.5): end);

    % training and testing sets
	traData_poisson = sparse(row_poisson(traIdx_poisson), col_poisson(traIdx_poisson), val_poisson(traIdx_poisson));
	tstData_poisson = sparse(row_poisson(tstIdx_poisson), col_poisson(tstIdx_poisson), val_poisson(tstIdx_poisson));

    %% lambdas
     datatypes_poisson = {{'poisson', 1.}};
     n_cols_poisson = [1000];
    [~, grad_source_poisson] = LikelihoodAndGradLikelihood(source_poisson, source_poisson, datatypes_poisson, n_cols_poisson);
    [~, lambda_poisson, ~] = lansvd(grad_source_poisson, 1);
   
    %% Optimization
	disp('%------------------------------------------------------------%');
	disp('%-                        PLAIS-Impute                      -%');
	disp('%------------------------------------------------------------%');
	 
%     para.test.row  = row(tstIdx);
% 	para.test.col  = col(tstIdx);
% 	para.test.data = val(tstIdx);
    para_poisson.test.test = false;
	para_poisson.maxIter = 50;
	para_poisson.tol = 1e-6;
    
	para_poisson.decay = 3.54e-1; % 3.4498e-1; % 3.123e-1;%3.445e-1;%3.5e-1; %2.5e-1;% 319e-1; %4e-1; %3.8e-1; %3.6e-1%4e-1;
    % the ideal is 3.5e-1 for the dataset whole 3.48% 3.44
    % 3.45e-1; %  gaussian
	para_poisson.exact = 0;
	para_poisson.maxR = 5*rank;
    
	% 
     [U_poisson, S_poisson, V_poisson, output_poisson ] = AccSoftImputeFastProx(traData_poisson, lambda_poisson, para_poisson, datatypes_poisson, n_cols_poisson );
     
     
	 sln_AISoftImpute_poisson = U_poisson *  S_poisson * V_poisson';
     obj_poisson{end+1} = output_poisson.obj;
     RanksIn_poisson{end+1} = output_poisson.RankIn;
     RanksOut_poisson{end+1} = output_poisson.RankOut;
     RMSE_poisson{end+1} = output_poisson.RMSE;
     Time_poisson{end+1} = output_poisson.Time;
     lambdas_poisson{end+1} = output_poisson.lambdas;
     recovered_data_poisson = zeros(length(row_poisson(tstIdx_poisson)),1);
     rows_poisson = [row_poisson(tstIdx_poisson)];
     cols_poisson = [col_poisson(tstIdx_poisson)];
     for k = 1:length(rows_poisson)
         recovered_data_poisson(k) = sln_AISoftImpute_poisson(rows_poisson(k), cols_poisson(k));
     end
     sln_AISoftImpute_tst_poisson = sparse(rows_poisson, cols_poisson, recovered_data_poisson, 3000, 1000);
     RMSE_montecarlo_poisson(m) = MatCompRMSERMSE(tstData_poisson, sln_AISoftImpute_tst_poisson);
 
	m = m +1;
    %RMSE_montecarlo_poisson(m) = MatCompRMSERMSE(tstData_poisson, sln_AISoftImpute_poisson);
end
fprintf('Mean of the montecarlo\n');
disp(mean(RMSE_montecarlo));
fprintf('std of the montecarlo\n');
disp(std(RMSE_montecarlo));
% fprintf("\n\n");
fprintf('Mean of the montecarlo in Poisson\n');
disp(mean(RMSE_montecarlo_poisson));
fprintf('std of the montecarlo in Poisson\n');
disp(std(RMSE_montecarlo_poisson));


  error('stop');
 %% Plots

figure
box on 
hold on;
ylim auto;
markersizes = ['d', 'o', 'd'];
colors = ['b', 'g', 'r'];

markersizesS = ['+', '*', 'x'];
colorsS = ['c', 'y', 'w'];

 for m=1:1
     RMSEin = RMSE{m};
     RMSEin  = RMSEin (1:length(RMSEin )-1);
     
    Timein = Time{m};
    Timein  = Timein(1: length(Timein)-1);
	plot(Timein, RMSEin , 'Marker', markersizes(m), 'color', colors(m), 'linewidth', 2, 'MarkerSize', 2);
    xlim([5 max(Timein)]);
 end 
xlabel('Time (seconds)');
ylabel('RMSE on training set');
title('Dimensions $$d_u = D = 3000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'rmse_3000_by_time.pdf');
close all;

figure
box on 
hold on;
ylim auto;
 for m=1:1
     RMSEin = RMSE{m};
     RMSEin  = RMSEin(1:length(RMSEin )-1);
     
    Timein = Time{m};
    Timein  = Timein(1: length(Timein)-1);
	plot(1: length(Timein), RMSEin , 'Marker', markersizes(m), 'color', colors(m), 'linewidth', 2, 'MarkerSize', 2);
    xlim([1 length(Timein)])
 end 
xlabel('Iterations');
ylabel('RMSE on training set');
title('$$d_u = D = 3000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'rmse_3000_by_iterations.pdf');
close all;

figure
box on;
hold on;
ylim auto;

for m=1:1
     objin = obj{m};
     objin  = objin(1:length(objin )-1);
     
    Timein = Time{m};
    Timein  = Timein(1: length(Timein)-1);
    
	%plot(Timein, objin , 'Marker', markersizes(m), 'color', colors(m), 'linewidth', 2, 'MarkerSize', 2);
    plot(Timein , objin , 'Marker', 'd', 'color', 'magenta', 'linewidth', 2, 'MarkerSize', 2);
    xlim([5 max(Timein)]);
end

xlabel('Time (seconds)');
ylabel('Objective $$\mathcal{F}_{\lambda}$$', 'Interpreter', 'Latex');
title('$$d_u = D = 3000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'objective_3000_by_time.pdf');
close all;

figure
box on;
hold on;
ylim auto;
% axis tight;

for m=1:1
     objin = obj{m};
     objin  = objin(1:length(objin )-1);
     
    Timein = Time{m};
    Timein  = Timein(1: length(Timein)-1);
    
	%plot(Timein, objin , 'Marker', markersizes(m), 'color', colors(m), 'linewidth', 2, 'MarkerSize', 2);
    plot(1:length(Timein) , objin , 'Marker', 'd', 'color', 'magenta', 'linewidth', 2, 'MarkerSize', 2);
    xlim([1 length(Timein)])
end

xlabel('Iterations');
ylabel('Objective $$\mathcal{F}_{\lambda}$$', 'Interpreter', 'Latex');
title('$$d_u = D = 3000$$', 'Interpreter', 'Latex');
saveas(gcf,'objective_3000_by_iterations.pdf');
close all;

figure
box on;
hold on;
%ylim auto;
%xlim auto;
axis tight;
for m=1:1
     objin = obj{m};
     objin  = objin (1:length(objin )-1);
     
    lambdasin = lambdas{m};
    lambdasin  = lambdasin(1: length(obj{m})-1);
    %disp(lambdasin);
	%plot(Timein, objin , 'Marker', markersizes(m), 'color', colors(m), 'linewidth', 2, 'MarkerSize', 2);
    plot(-log(lambdasin), objin , 'Marker', 'd', 'color', 'magenta', 'linewidth', 2, 'MarkerSize', 2);
    % xlim([1./max(lambdasin) 1./min(lambdasin)]);
    %plot(1:length(obj{m}) -1, lambdasin, 'Marker', 'd', 'color', 'magenta', 'linewidth', 2, 'MarkerSize', 2);
end

xlabel('$$-\log(\lambda)$$', 'Interpreter', 'Latex');
ylabel('Objective $$\mathcal{F}_{\lambda}$$', 'Interpreter', 'Latex');
title('$$d_u = D = 3000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'objective_3000_by_lambdas_lambdas.pdf');
close all;

figure
box on;
hold on;
ylim auto;
for m=1:1
    Timein = Time{m};
    RanksInin = RanksIn{m};
    RanksOutin = RanksOut{m};
    RanksInin = RanksInin(1: length(Timein) -1);
    RanksOutin = RanksOutin (1: length(Timein) -1);
    plot(1:length(Timein)-1, RanksInin, 'Marker', 'p', 'color', 'g',  'linewidth', 2, 'MarkerSize', 2);
	plot(1:length(Timein)-1, RanksOutin, 'Marker', 's', 'color', 'c',  'linewidth', 2, 'MarkerSize', 2);
    xlim([1 length(Timein)-1]);
    legend({'RankIn' 'RankOut'},'FontSize',10);
   
end
xlabel('Iterations');
ylabel('Learning ranks');
title('$$d_u = D = 3000$$', 'Interpreter', 'Latex');
saveas(gcf,'ranks_3000.pdf');
close all;