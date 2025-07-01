% 清除工作区变量、命令窗口内容，关闭所有图形窗口，并关闭警告信息
clear;clc;close all;warning off
% 将当前文件夹及其子文件夹添加到 MATLAB 的搜索路径中，以便可以调用其中的函数
addpath(genpath(pwd));

%% 对前10个优化函数进行循环处理
for i = 1:10
    % 将整数 i 转换为字符串
    F = sprintf('%d', i);
    % 生成函数名，格式为 'F' 加上数字，例如 'F1', 'F2' 等
    Function_name = ['F', F]; 
    % 调用 Get_Functions_details_1 函数，获取当前所选函数的相关信息
    % lb: 变量的下界
    % ub: 变量的上界
    % dim: 变量的维度
    % fobj: 目标函数句柄
    [lb, ub, dim, fobj] = Get_Functions_details_1(Function_name); 
    
    % 设置种群数量，即搜索代理的数量
    nPop = 30; 
    % 设置最大迭代次数，算法将在达到此迭代次数后停止
    Max_iter = 1000; 

    %% 调用算法
    %% 初始化一个空的元胞数组，用于保存优化结果
    Optimal_results = {}; 
    % 初始化结果索引
    index = 1; 

    % 开始计时，用于记录算法的运行时间
    tic
    % 定义优化算法的名称为 "GWO"（灰狼优化算法）
    optimization_name = 'GWO'; 
    % 调用 GWO 函数进行优化，传入种群数量、最大迭代次数、上下界、维度和目标函数句柄
    % Best_score: 找到的最优函数值
    % Best_x: 对应的最优变量
    % cg_curve: 收敛曲线，记录每次迭代的最优函数值
    [Best_score, Best_x, cg_curve] = GWO(nPop, Max_iter, lb, ub, dim, fobj);
    % 将算法名称保存到结果数组中
    Optimal_results{1, index} = optimization_name; 
    % 将收敛曲线保存到结果数组中
    Optimal_results{2, index} = cg_curve; 
    % 将最优函数值保存到结果数组中
    Optimal_results{3, index} = Best_score; 
    % 将最优变量保存到结果数组中
    Optimal_results{4, index} = Best_x; 
    % 停止计时，并将运行时间保存到结果数组中
    Optimal_results{5, index} = toc; 
    % 索引加 1，为下一个结果做准备
    index = index + 1; 

    % 显示信息，输出使用 GWO 算法为当前函数找到的最优目标函数值
    display(['The best optimal value of the objective funciton found by ', [num2str(optimization_name)], ' for ', [num2str(Function_name)], '  is : ', num2str(Best_score)]);   

    %% 绘制收敛曲线和目标函数的三维图
    % 创建一个新的图形窗口，并设置其位置和大小
    figure; set(gcf, 'position', [300, 300, 800, 330])
    % 将图形窗口划分为 1 行 2 列的子图，并选择第一个子图
    subplot(1, 2, 1);
    % 调用 func_plot_1 函数，绘制当前目标函数的三维图
    func_plot_1(Function_name)
    % 设置子图的标题为当前函数名
    title(Function_name)
    % 设置 x 轴标签
    xlabel('x')
    % 设置 y 轴标签
    ylabel('y')
    % 设置 z 轴标签
    zlabel('z')

    % 选择第二个子图
    subplot(1, 2, 2);
    % 遍历结果数组中的每一个结果
    for i = 1:size(Optimal_results, 2)
        % 使用半对数坐标绘制收敛曲线，设置线宽为 2
        semilogy(Optimal_results{2, i}, 'Linewidth', 2)
        % 保持图形，以便在同一图中绘制多条曲线
        hold on
    end
    % 设置子图的标题为当前函数名
    title(Function_name)
    % 设置 x 轴标签为迭代次数
    xlabel('Iteration');
    % 设置 y 轴标签为当前函数的最优得分
    ylabel(['Best score on ', num2str(Function_name)]);
    % 调整坐标轴范围，使图形紧凑
    axis tight
    % 显示网格线
    grid on;
    % 显示图形边框
    box on
    % 显示图例，标注每条曲线对应的算法名称
    legend(Optimal_results{1, :})
end

% 将之前添加到搜索路径的文件夹及其子文件夹从路径中移除，恢复原始路径设置
rmpath(genpath(pwd));