clear;close;clc;
set(0,'DefaultAxesFontSize',24,'DefaultAxesFontName','Arial');
set(0,'DefaultTextFontSize',24,'DefaultTextFontName','Arial');

%% 

N = 5; %choose between [5, 10, 20]


bare = load(['bare_data_' num2str(N) '.txt']);
full_data = load(['full_data_' num2str(N) '.txt']);

circuit_bare_ZZ = load(['circuit_ZZ_exp_' num2str(N) '_bare.txt']);
circuit_opt_ZZ = load(['circuit_ZZ_exp_' num2str(N) '_opt.txt']);

circuit_bare_O = load(['circuit_O_exp_' num2str(N) '_bare.txt']);
circuit_opt_O = load(['circuit_O_exp_' num2str(N) '_opt.txt']);

fidelity_bare = load(['circuit_fidelity_with_GS_' num2str(N) '_bare.txt']);
fidelity_opt = load(['circuit_fidelity_with_GS_' num2str(N) '_opt.txt']);

N_g = 50;
g_vals = linspace(-1, 1, N_g + 2);
g_vals = g_vals(2:end-1);


figure('Position', [10 10 1100 600])
hold on
plot(bare(:, 1), bare(:, 2),'-','LineWidth',3,'Color',[116,173,209]/255,'MarkerSize',8)
plot(full_data(:, 1), full_data(:, 2),'--x','LineWidth',1,'Color',[116,173,209]/255,'MarkerSize',8)

plot(g_vals, circuit_bare_O,'-s','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
plot(g_vals, circuit_opt_O,'--d','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
xlabel('$g$','Interpreter','latex')
ylabel('Expectation vals','Interpreter','latex')
title(['Number of spins = ' num2str(N)])
l = legend('DMRG bare', 'DMRG optimized', 'circuit bare', 'circuit optimized');
set(l,'Interpreter','latex')
ylim([0 1])
xlim([-1 1])
box on
ax = gca;
ax.XColor = 'k';
ax.YColor = 'k';
ax.TickLength = [0.015 0.01];
ax.TickLabelInterpreter = 'latex';
ax.LineWidth = 2;
% ax.FontWeight = 'normal';
set(gca, 'FontName', 'Arial')


figure('Position', [10 10 1100 600])
hold on
plot(bare(:, 1), bare(:,3),'-','LineWidth',3,'Color',[116,173,209]/255,'MarkerSize',8)
plot(full_data(:, 1), full_data(:, 3),'--x','LineWidth',1,'Color',[116,173,209]/255,'MarkerSize',8)
plot(g_vals, circuit_bare_ZZ / (N - 1),'-s','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
plot(g_vals, circuit_opt_ZZ / (N - 1),'--d','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
xlabel('$g$','Interpreter','latex')
ylabel('Expectation vals','Interpreter','latex')
title(['Number of spins = ' num2str(N)])
l = legend('DMRG bare', 'DMRG optimized', 'circuit bare', 'circuit optimized');
set(l,'Interpreter','latex')
ylim([0 1])
xlim([-1 1])
box on
ax = gca;
ax.XColor = 'k';
ax.YColor = 'k';
ax.TickLength = [0.015 0.01];
ax.TickLabelInterpreter = 'latex';
ax.LineWidth = 2;
% ax.FontWeight = 'normal';
set(gca, 'FontName', 'Arial')



figure('Position', [10 10 1100 600])
hold on
plot(g_vals, fidelity_bare,'-s','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
plot(g_vals, fidelity_opt,'--d','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
xlabel('$g$','Interpreter','latex')
ylabel('Expectation vals','Interpreter','latex')
title(['Number of spins = ' num2str(N)])
l = legend('fidelity with GS bare', 'fidelity with GS optimized');
set(l,'Interpreter','latex')
ylim([0.84 1])
xlim([-1 1])
box on
ax = gca;
ax.XColor = 'k';
ax.YColor = 'k';
ax.TickLength = [0.015 0.01];
ax.TickLabelInterpreter = 'latex';
ax.LineWidth = 2;
% ax.FontWeight = 'normal';
set(gca, 'FontName', 'Arial')