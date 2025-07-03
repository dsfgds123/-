% main_compare_GWO_vs_AGGWO.m
% =========================================================================
% 本腳本旨在嚴格對比原始GWO與我們改進的A-GGWO在GNSS整周模糊度
% 求解問題上的性能表現。
% =========================================================================

clc
clear
close all

%% ==================== 1. 數據載入與預處理 (公平的起點) ====================
load('small.mat'); %small.mat，sixdim.mat，large.mat

disp('數據載入與LAMBDA降相關...');
[L2, D2] = Chelosky(Q);
[Z1, L1, D1] = reduction(L2, D2);
a1 = (Z1.' * a)';
Q1 = Z1.' * Q * Z1;
disp('預處理完成！');
disp('------------------------------------------');

%% ==================== 2. 演算法公共參數設置 ====================
SearchAgents_no = 30;
Max_iter = 100;
dim = size(a, 1);

search_center = round(a1);
search_radius = 6;
lb = search_center - search_radius;
ub = search_center + search_radius;

% **重要：使用整數評估的目標函數**
fobj = @(x) target_function(a1, Q1, round(x)); % 搜索時保持連續，評估時使用整數

%% ==================== 3. 運行原始 GWO 進行基準測試 ====================
disp('正在運行原始 GWO 演算法...');
tic;
% 注意：我們需要一個GWO的整數優化版本，這裡直接修改調用方式
[gwo_score, gwo_pos, gwo_curve] = GWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj);
gwo_time = toc;
disp(['原始 GWO 完成！ 耗時: ', num2str(gwo_time), 's, 最優值: ', num2str(gwo_score)]);

%% ==================== 4. 運行我們改進的 A-GGWO =====================
disp('正在運行我們改進的 A-GGWO 演算法...');
AGGWO_params.s_max = 10;
AGGWO_params.Pm = 0.1;
AGGWO_params.Qn = Q1; % 傳入物理信息

tic;
[aggwo_score, aggwo_pos, aggwo_curve] = A_GGWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj, AGGWO_params);
aggwo_time = toc;
disp(['A-GGWO 完成！ 耗時: ', num2str(aggwo_time), 's, 最優值: ', num2str(aggwo_score)]);
disp('------------------------------------------');

%% ==================== 5. 結果可視化對比 ====================
figure('Name', 'GWO vs A-GGWO Performance Comparison');
semilogy(gwo_curve, 'r-', 'LineWidth', 2);
hold on;
semilogy(aggwo_curve, 'b--', 'LineWidth', 2.5);
title('GWO vs. A-GGWO 在GNSS模糊度搜索上的收斂曲線對比');
xlabel('迭代次數 (Iteration)');
ylabel('最優適應度值 (Best Fitness)');
legend('原始 GWO (Original GWO)', '我們的 A-GGWO (Our A-GGWO)');
grid on;
box on;
axis tight;

disp('對比實驗完成！請查看生成的收斂曲線對比圖。');