% clear;close;clc;
% set(0,'DefaultAxesFontSize',24,'DefaultAxesFontName','Arial');
% set(0,'DefaultTextFontSize',24,'DefaultTextFontName','Arial');
% 
% 
% w_5 = load('bar_vals_5.txt');
% w_10 = load('bar_vals_10.txt');
% w_20 = load('bar_vals_20.txt');
% 
% 
% 
% figure('Position', [10 10 1600 600])
% xlabel('Coefficient index')
% ylabel('Weight')
% title('Coefficient Relevance Weights')
% box on
% set(gca, 'FontName', 'Arial', 'LineWidth', 1.5)
% ylim([0 0.35])
% hold on
% labels = {'XX', 'YY', 'XIX', 'YIY', 'ZIZ',...
%     'XXX', 'YXY',...
%     'YZZY', 'ZYYZ', 'YXXY', 'YYYY',...
%     'ZXXZ', 'ZZZZ',...
%     'XXXX', 'XYYX', 'XZZX'}; 
% bar(w_5, 'FaceColor', [166,206,227]/255)
% bar(w_10, 'FaceColor', [178,223,138]/255)
% bar(w_20, 'FaceColor', [202,178,214]/255)
% set(gca, 'XTick', 1:16, 'XTickLabel', labels, 'XTickLabelRotation', 45)
% ylabel('Weight')
% title('Operator Weights')

clear; close all; clc;

% Consistent typography
set(0,'DefaultAxesFontSize',24,'DefaultAxesFontName','Arial',...
      'DefaultTextFontSize',24,'DefaultTextFontName','Arial');

% Load data (each should be a 16x1 vector)
w_5  = load('bar_vals_5.txt');
w_10 = load('bar_vals_10.txt');
w_20 = load('bar_vals_20.txt');

labels = {'XX','YY','XIX','YIY','ZIZ',...
          'XXX','YXY',...
          'YZZY','ZYYZ','YXXY','YYYY',...
          'ZXXZ','ZZZZ',...
          'XXXX','XYYX','XZZX'};

W = [w_5(:) w_10(:) w_20(:)];   % 16 x 3 -> grouped bars

figure('Position',[10 10 1200 600],'Color','w');

% Grouped bars, narrower width for clear separation
b = bar(W,'grouped','BarWidth',0.65);  % try 0.55–0.70 to taste

% Colors that print well and are colorblind-friendly
cols = [166 206 227;   % light blue
        178 223 138;   % light green
        202 178 214]/255; % light purple
for i = 1:numel(b)
    b(i).FaceColor = cols(i,:);
    b(i).EdgeColor = 'k';
    b(i).LineWidth = 0.8;
    b(i).FaceAlpha = 0.95;
end

ax = gca; box on; hold on;
ax.LineWidth = 1.5;
ax.TickLabelInterpreter = 'latex';
ax.XTick = 1:numel(labels);
ax.XTickLabel = labels;
ax.XTickLabelRotation = 35;
ax.YLim = [0 0.35];
ax.XLim = [0.5 numel(labels)+0.5];
ax.TickLength = [0.015 0.01];

xlabel('$\text{Operator index}$','Interpreter','latex');
ylabel('Coefficient Relevance Weights','Interpreter','latex');


% Legend with LaTeX and no box
lg = legend({'$N=5$','$N=10$','$N=20$'},...
            'Interpreter','latex','Location','northwest');
set(lg,'Box','off');
% Move legend slightly down (negative y-shift)
pos = lg.Position;       % [x y width height]
pos(1) = pos(1) + 0.02;   % shift right
pos(2) = pos(2) - 0.03;  % shift down by 3% of figure height
lg.Position = pos;

% Optional: thin gridlines to help compare heights without clutter
grid on; ax.GridAlpha = 0.2; ax.GridColor = [0 0 0];
exportgraphics(gcf,'operator_weights.pdf','ContentType','vector');

