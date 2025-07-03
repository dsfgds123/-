% % run_AGGWO_on_GNSS.m 
% % =========================================================================
% clc
% clear
% close all
% 
% %% ==================== 1. 数据加载与预处理 ====================
% load('sixdim.mat'); %%small.mat，sixdim.mat，large.mat
% 
% disp('数据加载完毕，开始进行LAMBDA降相关...');
% [L2, D2] = Chelosky(Q);
% [Z1, L1, D1] = reduction(L2, D2);
% a1 = (Z1.' * a)';
% Q1 = Z1.' * Q * Z1;
% disp('降相关完成！');
% disp('------------------------------------------');
% 
% 
% %% ==================== 2. A-GGWO 算法参数设置 ====================
% disp('正在设置 A-GGWO 算法参数...');
% 
% SearchAgents_no = 30;
% Max_iter = 100;
% dim = size(a, 1);
% 
% search_center = round(a1);
% % --- 方案二：新的搜索边界设置 ---覆盖高概率区域的搜索半径。
% % % 1. 从降相关后的协方差矩阵Q1计算每个维度的标准差
% % std_devs = sqrt(diag(Q1)); 
% % 
% % % 2. 定义一个缩放因子（例如3，代表3-sigma）
% % scaling_factor = 3; 
% % 
% % % 3. 计算每个维度独立的半径
% % radius_vector = scaling_factor * std_devs;
% % 
% % % 4. 为了保证所有个体都在一个超立方体内，可以取所有维度半径中的最大值，并向上取整
% % %    这确保了搜索立方体能包裹住真实的置信椭球的主要部分。
% % search_radius = ceil(max(radius_vector));
% % disp(['根据数据统计特性，动态设置搜索半径为: ', num2str(search_radius)]);
% % 
% % % 5. 设置搜索边界
% % lb = search_center - search_radius;
% % ub = search_center + search_radius;
% 
% search_radius = 6;
% lb = search_center - search_radius;
% ub = search_center + search_radius;
% 
% fobj = @(x) target_function(a1, Q1, round(x));
% 
% AGGWO_params.s_max = 10;
% AGGWO_params.Pm = 0.1;
% AGGWO_params.Qn = Q1;
% 
% disp('参数设置完毕，准备启动 A-GGWO 进行搜索...');
% disp('------------------------------------------');
% 
% %% ==================== 3. 调用 A-GGWO 核心算法 ====================
% tic;
% % 使用我们最新的、能返回正确整数解的 A_GGWO (v2.1)
% [Best_pos, Best_score, Convergence_curve] = A_GGWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj, AGGWO_params);
% execution_time = toc;
% 
% %% ==================== 4. 结果展示 (已修正) ====================
% % Best_pos 现在直接是整数解了
% Final_solution_Z = Best_pos; 
% % 从Z空间反算回原始空间
% Final_solution_Orig = inv(Z1) * Final_solution_Z'; 
% 
% % --- 修正开始 ---
% disp('搜索完成！');
% disp(['算法运行时间: ', num2str(execution_time), ' 秒']);
% 
% % 正确显示 Best_score (一个标量)
% disp(['找到的最小目标函数值 (在Z空间): ', num2str(Best_score)]);
% 
% % 正确显示 Best_pos (一个向量)
% % 使用 sprintf 来格式化向量，避免维度问题
% z_space_str = sprintf('%d ', Final_solution_Z);
% disp(['在Z空间找到的最优整数解 (Best_pos): [', z_space_str, ']']);
% 
% orig_space_str = sprintf('%.4f ', Final_solution_Orig');
% disp(['转换回原始空间的最终整数解: [', orig_space_str, ']']);
% % --- 修正结束 ---
% 
% %% ==================== 5. 绘图 ====================
% figure;
% plot(Convergence_curve, 'b-', 'LineWidth', 2);
% title('A-GGWO on GNSS Ambiguity Search');
% xlabel('迭代次数 (Iteration)');
% ylabel('最优适应度值 (Best Fitness)');
% grid on;
% box on;
% run_AGGWO_on_GNSS.m 
% =========================================================================
clc
clear
close all

%% ==================== 1. 数据加载与预处理 ====================
load('sixdim.mat'); % 您可以在这里切换 'small.mat', 'sixdim.mat', 'large.mat'

disp('数据加载完毕，开始进行LAMBDA降相关...');
[L2, D2] = Chelosky(Q);
[Z1, L1, D1] = reduction(L2, D2);
a1 = (Z1.' * a)';
Q1 = Z1.' * Q * Z1;
disp('降相关完成！');
disp('------------------------------------------');


%% ==================== 2. A-GGWO 算法参数设置 ====================
disp('正在设置 A-GGWO 算法参数...');

SearchAgents_no = 30;
Max_iter = 1000;
dim = size(a, 1);

search_center = round(a1);
% --- 方案二：新的搜索边界设置 ---覆盖高概率区域的搜索半径。
% 1. 从降相关后的协方差矩阵Q1计算每个维度的标准差
std_devs = sqrt(diag(Q1)); 

% 2. 定义一个缩放因子（例如3，代表3-sigma）
scaling_factor = 3; 

% 3. 计算每个维度独立的半径
radius_vector = scaling_factor * std_devs;

% 4. 为了保证所有个体都在一个超立方体内，可以取所有维度半径中的最大值，并向上取整
%    这确保了搜索立方体能包裹住真实的置信椭球的主要部分。
search_radius = max(6, ceil(max(radius_vector)));
disp(['根据数据统计特性，动态设置搜索半径为: ', num2str(search_radius)]);

% 5. 设置搜索边界 (lb和ub本身就是整数)
lb = search_center - search_radius;
ub = search_center + search_radius;

% --- MODIFICATION START ---
% MODIFICATION: 移除了这里的round()。我们让A_GGWO算法内部保证传入的x是整数。
% 原始代码: fobj = @(x) target_function(a1, Q1, round(x));
fobj = @(z) target_function(a1, Q1, z); % z现在被假定为整数向量
% --- MODIFICATION END ---

AGGWO_params.s_max = 10;
AGGWO_params.Pm = 0.1;
AGGWO_params.Qn = Q1;

disp('参数设置完毕，准备启动 A-GGWO 进行搜索...');
disp('------------------------------------------');

%% ==================== 3. 调用 A-GGWO 核心算法 ====================
tic;
% 调用修改后的 A_GGWO 函数
[Best_score, Best_pos, Convergence_curve] = A_GGWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj, AGGWO_params);
execution_time = toc;

%% ==================== 4. 结果展示 ====================
Final_solution_Z = Best_pos; 
% 从Z空间反算回原始空间
% 注意：要确保Z1非奇异。对于LAMBDA方法，Z1的行列式为±1，一定是可逆的。
Final_solution_Orig = Z1 \ Final_solution_Z'; % 使用更稳健的左除法 inv(Z1)*...

disp('搜索完成！');
disp(['算法运行时间: ', num2str(execution_time), ' 秒']);
disp(['找到的最小目标函数值 (在Z空间): ', num2str(Best_score)]);

% 使用更可靠的方式来显示向量
z_space_str = sprintf('%d ', Final_solution_Z);
disp(['在Z空间找到的最优整数解 (Best_pos): [', z_space_str, ']']);

orig_space_str = sprintf('%.4f ', Final_solution_Orig);
disp(['转换回原始空间的最终整数解: [', orig_space_str, ']']);

%% ==================== 5. 绘图 ====================
figure;
plot(Convergence_curve, 'b-', 'LineWidth', 2);
title('A-GGWO on GNSS Ambiguity Search (Corrected)');
xlabel('迭代次数 (Iteration)');
ylabel('最优适应度值 (Best Fitness)');
grid on;
box on;