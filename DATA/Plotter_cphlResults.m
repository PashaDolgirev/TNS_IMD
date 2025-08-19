clear;close;clc;
set(0,'DefaultAxesFontSize',24,'DefaultAxesFontName','Arial');
set(0,'DefaultTextFontSize',24,'DefaultTextFontName','Arial');

%% 

N = 20; %choose between [5, 10, 20]


bare = load(['bare_data_' num2str(N) '.txt']);
full_data = load(['full_data_' num2str(N) '.txt']);


% figure('Position', [10 10 1100 600])
% hold on
% plot(bare(:, 1), bare(:, 2),'-','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
% plot(full_data(:, 1), full_data(:, 2),'--x','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
% plot(bare(:, 1), bare(:,3),'-','LineWidth',1,'Color',[116,173,209]/255,'MarkerSize',8)
% plot(full_data(:, 1), full_data(:, 3),'--x','LineWidth',1,'Color',[116,173,209]/255,'MarkerSize',8)
% xlabel('$g$','Interpreter','latex')
% ylabel('Expectation vals','Interpreter','latex')
% title(['Number of spins = ' num2str(N)])
% ylim([0 1])
% xlim([-1 1])
% box on
% ax = gca;
% ax.XColor = 'k';
% ax.YColor = 'k';
% ax.TickLength = [0.015 0.01];
% ax.TickLabelInterpreter = 'latex';
% ax.LineWidth = 2;
% % ax.FontWeight = 'normal';
% set(gca, 'FontName', 'Arial')


figure('Position', [10 10 1100 600])
hold on
plot(bare(:, 1), bare(:, 2),'-','LineWidth',3,'Color',[165,0,38]/255,'MarkerSize',8)
plot(bare(:, 1), bare(:,3),'-','LineWidth',3,'Color',[116,173,209]/255,'MarkerSize',8)
plot(full_data(:, 1), full_data(:, 2),'--x','LineWidth',1,'Color',[165,0,38]/255,'MarkerSize',8)
plot(full_data(:, 1), full_data(:, 3),'--x','LineWidth',1,'Color',[116,173,209]/255,'MarkerSize',8)
xlabel('$g$','Interpreter','latex')
ylabel('Expectation vals','Interpreter','latex')
title(['Number of spins = ' num2str(N)])
l = legend('$\langle {\cal O} \rangle$', '$\langle Z_i Z_{i + 1} \rangle$');
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

%%

CMat = load(['CMat_' num2str(N) '.txt']);

w = max(abs(CMat));%encodes weight of the given Hamiltonian parameter
w_norm = (w - min(w)) / (max(w) - min(w) + eps);  
cmap = hot(256);
color_indices = round(1 + w_norm * 0.3 * (size(cmap,1)-1));  % indices in [1, 256]
colors = cmap(color_indices, :);

g_vals = bare(:, 1);

figure('Position', [10 10 500 900])
hold on
% Plot each coefficient curve with its corresponding color
for i = 1:size(CMat, 2)
    plot(g_vals, CMat(:, i), '-', 'LineWidth', 2, 'Color', colors(i, :))
end


xlabel('$g$','Interpreter','latex')
ylabel('Coefficients','Interpreter','latex')
title(['Number of spins = ' num2str(N)])
ylim([-0.1 0.35])
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


writematrix(w, ['bar_vals_' num2str(N) '.txt'])

%%


figure('Position', [10 10 1600 600])
xlabel('Coefficient index')
ylabel('Weight')
title('Coefficient Relevance Weights')
box on
set(gca, 'FontName', 'Arial', 'LineWidth', 1.5)
ylim([0 0.35])

labels = {'X_{i}X_{i+1}', 'Y_{i}Y_{i+1}', 'X_{i}X_{i+2}', 'Y_{i}Y_{i+2}', 'Z_{i}Z_{i+2}',...
    'X_{i - 1} X_{i} X_{i + 1}', 'Y_{i - 1} X_{i} Y_{i + 1}',...
    'Y_{i - 1} Z_{i} Z_{i + 1} Y_{ i + 2 }', 'Z_{i - 1} Y_{i} Y_{i + 1} Z_{ i + 2 }', 'Y_{i - 1} X_{i} X_{i + 1} Y_{ i + 2 }', 'Y_{i - 1} Y_{i} Y_{i + 1} Y_{ i + 2 }',...
    'Z_{i - 1} X_{i} X_{i + 1} Z_{ i + 2 }', 'Z_{i - 1} Z_{i} Z_{i + 1} Z_{ i + 2 }',...
    'X_{i - 1} X_{i} X_{i + 1} X_{ i + 2 }', 'X_{i - 1} Y_{i} Y_{i + 1} X_{ i + 2 }', 'X_{i - 1} Z_{i} Z_{i + 1} X_{ i + 2 }'}; 
% set(labels, 'Interpreter', 'latex')
bar(w, 'FaceColor', [0.2 0.6 0.5])
set(gca, 'XTick', 1:16, 'XTickLabel', labels, 'XTickLabelRotation', 45)
ylabel('Weight')
title('Operator Weights')
