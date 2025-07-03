% A_GGWO.m
% Version 3.0 - Core Mechanism Rebuilt for Integer Problems
% Replaces the standard GWO update with a diversity-preserving local search,
% inspired by the analysis of successful integer ambiguity solvers.

function [Alpha_score, Alpha_pos, Convergence_curve] = A_GGWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj, AGGWO_params)

% 初始化 Alpha, Beta, Delta 狼 (依然用于记录最优解)
Alpha_pos = zeros(1, dim);
Alpha_score = inf;
Beta_pos = zeros(1, dim);
Beta_score = inf;
Delta_pos = zeros(1, dim);
Delta_score = inf;

% 初始化狼群位置
Positions = initialization(SearchAgents_no, dim, ub, lb);
Positions = round(Positions); % 保证初始种群是整数

Convergence_curve = zeros(1, Max_iter);

% 主循环
for l = 1:Max_iter
    % 1. 评估和更新最优解记录 (Alpha, Beta, Delta)
    for i = 1:size(Positions, 1)
        % 边界检查
        Positions(i,:) = max(Positions(i,:), lb);
        Positions(i,:) = min(Positions(i,:), ub);
        
        fitness = fobj(Positions(i,:));
        
        if fitness < Alpha_score
            Delta_score = Beta_score; Delta_pos = Beta_pos;
            Beta_score = Alpha_score; Beta_pos = Alpha_pos;
            Alpha_score = fitness; Alpha_pos = Positions(i,:);
        elseif fitness < Beta_score
            Delta_score = Beta_score; Delta_pos = Beta_pos;
            Beta_score = fitness; Beta_pos = Positions(i,:);
        elseif fitness < Delta_score
            Delta_score = fitness; Delta_pos = Positions(i,:);
        end
    end
    
    % ======================== 核心算法重构 ========================
    % 抛弃标准GWO的强收敛更新规则，以避免种群暴毙。
    % 采用新的、保持多样性的“局部随机探索”策略。
    %==============================================================
    
    % 动态调整探索步长：从一个较大的值逐渐减小
    % 初始步长可以是搜索范围的10%，最终减小到一个很小的值
    max_step = max(1, floor(0.1 * (ub(1) - lb(1)))); % 初始最大步长
    step_size = max(1, round(max_step * (1 - l/Max_iter))); % 步长随迭代次数减小
    
    % Elitism: 保留当前最优的Alpha狼，不对它进行探索
    new_positions = Alpha_pos; 
    
    % 对其他所有狼进行独立的局部探索
    for i = 2:SearchAgents_no
        current_pos = Positions(i,:);
        
        % 生成一个随机的整数探索向量
        random_step = round(step_size * randn(1, dim));
        
        % 产生新的候选位置
        candidate_pos = current_pos + random_step;
        
        % (可选但推荐) 边界检查
        candidate_pos = max(candidate_pos, lb);
        candidate_pos = min(candidate_pos, ub);
        
        % 计算新位置的适应度
        candidate_fitness = fobj(candidate_pos);
        
        % 如果新位置更好，则接受它；否则，保留原位
        if candidate_fitness < fobj(current_pos)
            new_positions(i,:) = candidate_pos;
        else
            new_positions(i,:) = current_pos;
        end
    end
    
    Positions = new_positions; % 用新一代的位置更新整个种群

    % 记录当次迭代的最优值
    Convergence_curve(l) = Alpha_score;
    if mod(l, 10) == 0
        disp(['Iteration ', num2str(l), ', Step Size: ', num2str(step_size), ', Best Fitness: ', num2str(Alpha_score)]);
    end
end

end