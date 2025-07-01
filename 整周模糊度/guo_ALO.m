clc
clear

%% 录入数据
load('small.mat') % 正确数值0.2183
% load('sixdim.mat') % 正确数值0.1882
% load('large.mat') % 正确数值15.0166
% n=length(a);%n为模数

%设置参数
maxIter=40; % 最大迭代次数
pop =20; % 蚁狮种群数量
dim=size(a,1); % dim 问题维数
beta=1;%搜索步长

%% 降相关
[L2,D2]=Chelosky(Q);%LDL分解
[Z1,L1,D1]=reduction(L2,D2);%进行整数Z变换
a1=(Z1.'*a)';%整数变换后的实数模糊度向量a1
Q1=Z1.'*Q*Z1;%整数变换后的协方差矩阵Q1


% %模糊度浮点解a
% %a=[5.45,3.1, 2.97];
% a=[-4.57,10.02, 2.35];
A=round(a1);
% %协方差矩阵Q
% %Q=[6.29 5.978 0.544;5.978 6.292 2.34;0.544 2.34 6.288];
% Q=[4.476,0.334,0.230;0.334,1.146,0.082;0.230,0.082,0.626];


%% 搜索部分
%初始化搜索空间
searchspace=zeros(13,dim);

%构造搜索空间
for d=1:dim
    for k=1:13
        searchspace(k,d)=A(1,d)+k-7; %满足论文中构建搜索空间的方法:[N]-L/λ ≤ N ≤ [N]+L/λ
    end
end

%% 初始化蚁狮&蚂蚁位置
antlion_position = initialization(pop,dim,searchspace);
ants_position = initialization(pop,dim,searchspace);
Iter_curve = zeros(1, maxIter);
antlions_fitness = zeros(1, pop);
ants_fitness = zeros(1, pop);

%% 计算蚁狮适应度
antlions_fitness=zeros(pop,1);
for i = 1:pop
    antlions_fitness(i) = target_function(a1,Q1,antlion_position(i,:));
end
% 对蚁群适应度值排序
[sorted_antlion_fitness, sorted_indexes] = sort(antlions_fitness,'descend');
% 根据适应度值将蚁狮位置整理
for i = 1 : pop
    Sorted_antlions(i, :) = antlion_position(sorted_indexes(i), :);
end
% 精英蚁狮位置&适应度
Elite_antlion_position = Sorted_antlions(1,:);
Elite_antlion_fitness = sorted_antlion_fitness(1);
% 记录全局最优值
Best_pos = Elite_antlion_position;
Best_fitness = Elite_antlion_fitness;
% 记录当代蚁狮位置&精英蚁狮位置
History_pos{1} = Sorted_antlions;
History_best{1} = Elite_antlion_position;
Iter_curve(1) = Best_fitness;
Cur_iter = 2;

%% 迭代
tic

lb=min(ants_position,antlion_position); 
ub=max(ants_position,antlion_position);

while Cur_iter < maxIter + 1
    % 随机游走
    for i = 1:pop
        % 轮盘赌策略随机选择一个蚁狮
        Rolette_index = RouletteWheelSelection(1./sorted_antlion_fitness); % 取倒数，适应度小，被选择概率大
        % 计算围绕随机蚁狮游走的RA
        RA = Random_walk_around_antlion(dim, maxIter, lb, ub, Sorted_antlions(Rolette_index, :), Cur_iter);
        % 计算围绕精英蚁狮游走的RE
        RE = Random_walk_around_antlion(dim, maxIter, lb, ub, Elite_antlion_position(1,:), Cur_iter);
        % 计算蚂蚁的位置
        ant_position(i, :) = (RA(Cur_iter, :) + RE(Cur_iter, :)) / 2;
    end
    
    for i = 1 : pop
        % 蚂蚁边界检查
        ant_position(i, :) = BoundaryCheck(ant_position(i, :), ub, lb, dim);
        % 计算蚂蚁适应度值
        ants_fitness(i) = target_function(a1,Q1,ant_position(i, :));
    end
    
    % 合并蚁狮和蚂蚁的位置
    double_population = [Sorted_antlions; ant_position];
    double_fitness = [sorted_antlion_fitness;(ants_fitness)'];
    % 排序
    [double_fitness_sorted, newIndex] = sort(double_fitness,'descend');
    double_sorted_population = double_population(newIndex, :);
    % 取前pop种群数量作为新精英蚁狮
    antlions_fitness = double_fitness_sorted(1:pop);
    Sorted_antlions = double_sorted_population(1:pop,:);
    % 更新精英蚁狮
    if antlions_fitness(1) < Elite_antlion_fitness
        Elite_antlion_position = Sorted_antlions(1, :);
        Elite_antlion_fitness = antlions_fitness(1);
    end
    % 确保精英蚁狮在第一个位置
    Sorted_antlions(1,:) = Elite_antlion_position;
    antlions_fitness(1) = Elite_antlion_fitness;

    % 记录全局最优值
    Best_pos = Elite_antlion_position;
    Best_fitness = Elite_antlion_fitness;
    History_pos{Cur_iter} = Sorted_antlions;
    History_best{Cur_iter} = Elite_antlion_position;
    Iter_curve(Cur_iter) = Best_fitness;
    Cur_iter = Cur_iter + 1;
end

%将ture值转换到新空间中
final_truth=(Z1.'*truth)';

toc

%绘图
plot(antlions_fitness);
xlabel('迭代次数')
ylabel('适应度函数值')





