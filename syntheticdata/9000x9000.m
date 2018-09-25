clear;
clc;

m = 1; 
obj = {};
RanksIn = {};
RanksOut = {};
RMSE = {};
Time = {};
lambdas = {};
RMSE_montecarlo = zeros(1,1);


while m <= 1
    fprintf("Number of iteration in the montecarlo %d:\n", m);
 	datatypes = {{'gaussian', [1.,  .05]}, {'poisson', 1.}, {'bernoulli', 0.5}};
    % datatypes = {{'gaussian', [1.,  .05]}};
 	% datatypes = {{'poisson', 1.}};
 	% datatypes = {{'bernoulli', 0.5}};
    
    %% dimensions
	n_rows = 9000;
	n_cols = [3000, 3000, 3000];
    
    %% ranks
	rank = 15;
	ranks = [rank, rank, rank];% ];
   
    %% probas
	probas = [.15, .15, .15];    
    
    %% simulations
	[source, ~, ~, ~, ~, ~] = ...
	CollectMCsimu(n_rows, n_cols, ranks, probas, datatypes);
    source = source / max(source(:));
    source = sparse(source);
    disp(nnz(source)/(27*1e6));
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

    %% training and testing sets
	traData = sparse(row(traIdx), col(traIdx), val(traIdx));
	tstData = sparse(row(tstIdx), col(tstIdx), val(tstIdx));
    
    %% lambdas
    [~, grad_source] = LikelihoodAndGradLikelihood(source, source, datatypes, n_cols);
     [~, lambda, ~] = lansvd(grad_source, 1);
    %lambda =  75 * lambda;
    disp(lambda);
   
 
    %% Optimization
	disp('%------------------------------------------------------------%');
	disp('%-                        PLAIS-Impute                      -%');
	disp('%------------------------------------------------------------%');
	 
    para.test.row  = row(tstIdx);
	para.test.col  = col(tstIdx);
	para.test.data = val(tstIdx);
    para.test.test = false;
	para.maxIter = 53;
	para.tol = 1e-7;
    
	para.decay = 1.179e-1; 
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
     RMSE_montecarlo(m) = MatCompRMSERMSE(tstData, sln_AISoftImpute);
 
	m = m +1;
    RMSE_montecarlo(m) = MatCompRMSERMSE(tstData, sln_AISoftImpute);
end
fprintf('Mean of the montecarlo\n');
disp(mean(RMSE_montecarlo));
fprintf('std of the montecarlo\n');
disp(std(RMSE_montecarlo));


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
title('$$d_u = D = 9000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'rmse_9000_by_time.pdf');
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
title('$$d_u = D = 9000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'rmse_9000_by_iterations.pdf');
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
title('$$d_u = D = 9000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'objective_9000_by_time.pdf');
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
title('$$d_u = D = 9000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'objective_9000_by_iterations.pdf');
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
title('$$d_u = D = 9000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'objective_9000_by_lambdas_lambdas.pdf');
close all;

figure
box on;
hold on;
ylim auto;
for m=1:1
    Timein = Time{m};
    % ylim([ 30]);
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
title('$$d_u = D = 9000$$', 'Interpreter', 'Latex');
% legend({'m=1', 'm=2', 'm=3'},'FontSize',10);
saveas(gcf,'ranks_9000.pdf');
close all;