clear;
clc;
close all;

%% 录入数据
% load('small.mat') % 正确数值0.2183
% load('sixdim.mat') % 正确数值0.1882
load('large.mat') % 正确数值15.0166
% n=length(a);%n为模数

%设置参数
maxIter=15; % 最大迭代次数
pop =10; % 蚁狮种群数量
dim=size(a,1); % dim 问题维数

% elapsedTime=[1,100];
% a3=2;%混沌映射参数


beta=1;%搜索步长
%fobj 适应度函数 

% %模拟退火算法1
% cooling_rate=0.99;%模拟退火速率参数
% initial_temperature=100;%模拟退火初始温度

%模拟退火算法2



%% 降相关
[L2,D2]=Chelosky(Q);%LDL分解
[Z1,L1,D1]=reduction(L2,D2);%进行整数Z变换
a1=(Z1.'*a)';%整数变换后的实数模糊度向量a1
Q1=Z1.'*Q*Z1;%整数变换后的协方差矩阵Q1


% %模糊度浮点解a
% %a=[5.45,3.1, 2.97];
% a=[-4.57,10.02, 2.35];
A=round(a1);%向右取整ceil()；向左取整floor()，向0取整fix()
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

%保存最优解的变量
history_best_value=zeros(1,maxIter);
% save=zeros(maxIter,dim);

%初始化蚁狮个体位置
alos_position=zeros(pop,dim); %每一个蚁狮都要找到一个最优值
for n=1:pop
    for e=1:dim
    alos_position(n,e)=searchspace(randi(13),e); %随机将搜索空间里面的值赋给蚁狮个体
    end
end

%初始化最优解
alos_best=alos_position;

%创建每次迭代每一个蚁狮个体的位置变量和每次迭代相应的适应度值
history_solution=cell(1,maxIter+1);
inter_alos_position=zeros(pop,dim); %蚁狮个体的位置变量
history_j=cell(0,maxIter+1);
inter_value=zeros(pop,1); %迭代相应的适应度值
for f=1:maxIter
    history_solution{1,f+1}=inter_alos_position; %把每次蚁狮个体的位置变量保存到相应的元胞数组中
    history_j{1,f+1}=inter_value; %把每次迭代相应的适应度值保存到相应的元胞数组中
end

%计算初始每一个蚁狮的适应度最优值
j=zeros(pop,1);   
for m=1:pop
    j(m)=target_function(a1,Q1,alos_position(m,:)); %计算每一个蚁狮最优适应度存入j(m)中
end
history_j{1}=j;
history_solution{1,1}=alos_best;
history_best_value(1)=min(history_j{1});


truth_number=0;%统计迭代100次的正确值个数
% 开始迭代搜索
for q=1:100 %统计时间
tic;

lb=min(alos_position);
ub=max(alos_position);


% 设置默认选项
options = struct('T0',10,'alpha',0.6,'T_min',1e-2,...
                         'max_iter',10,'verbose',false);

%迭代次数
for inter=2:maxIter
    %更新每一个蚁狮的位置
    for i=1:pop   
        for k=1:dim
        %% 
            % 轮盘赌随机选择一个蚁狮
            % Rolette_index=RouletteWheelSelection(1./history_j{1}); % 取倒数，适应度小，被选择概率大
            % % 计算围绕随机蚁狮游走的RA
            % RA = Random_walk_around_antlion(dim, maxIter, lb, ub, alos_position(Rolette_index, :), inter);
            %  % 计算围绕精英蚁狮游走的RE
            % RE = Random_walk_around_antlion(dim, maxIter, lb, ub, alos_position(1,:), inter);

            % % 生成特殊蚁狮
            % rand = unidrnd(k,pop);
            % RS = obtain_spec_alos(rand, alos_position(1,:), dim,lb,ub);

             %logistic-tent混沌映射
             % alos_position(i, :) = logistic_tent_map(alos_position(i, :), r);

             %Tent混沌映射
            % H(i,:) = Tent_int(dim,a3,lb,ub);

            %PSA搜索算法改进步长
            % L=round(PSA_levyFlight(beta));
           

            % 计算蚁狮的位置()
            % 飞行步长系数
            % alos_position(i, :) = round((RA(inter, :) + RE(inter, :)) / 2);
            % alos_position(i, :) = round((RA(inter, :) + RE(inter, :)) / 2)+L;
            % alos_position(i, :) = round((RA(inter, :) + RE(inter, :)) / 2)+L+H(i,:)/3;
            % alos_position(i, :) = round((RA(inter, :) + RE(inter, :)+H(i,:)) / 3)+L;
            % alos_position(i, :) = alos_position(i, :)+L;

            %模拟退火算法1
            % best_position(i,:)=simulated_annealing(a1,Q1,alos_position(i,:), maxIter, initial_temperature, cooling_rate);

            %模拟退火算法2
            [best_position(i,:),fval{inter-1}(k,i)]=simulated_annealing1(a1,Q1,alos_position(i,:),options);
            alos_position(i, :) = best_position(i, :);   
            
            % L=round(PSA_levyFlight(beta));
            % alos_position(i,k)=alos_position(i,k)+L; 

            %每一个更新的蚁狮的位置限制范围
            for f=1:dim
            alos_position(i,f)=max(alos_position(i,f),min(searchspace(:,f)));
            % alos_position(i,2)=max(alos_position(i,2),min(searchspace(:,2)));
            % alos_position(i,3)=max(alos_position(i,3),min(searchspace(:,3)));
            % alos_position(i,1)=min(alos_position(i,1),max(searchspace(:,1)));   
            % alos_position(i,2)=min(alos_position(i,2),max(searchspace(:,2)));
            alos_position(i,f)=min(alos_position(i,f),max(searchspace(:,f)));
            end


        % %随机初始化蚁狮位置：rand(0,1).*(max-min)+min
        % alos_position=round(unifrnd(min(searchspace),max(searchspace)));
        % %计算适应度值
        %  j(i)=target_function(a,Q,alos_position(i,:));
        end
        J=target_function(a1,Q1,alos_position(i,:));
           history_j{1,inter}(i,1)=J;
        if history_j{1,inter}(i,1)<history_j{1,inter-1}(i,1)
            history_solution{1,inter}(i,:)=alos_position(i,:);
        else
            history_solution{1,inter}(i,:) =history_solution{1,inter-1}(i,:); 
            history_j{1,inter}(i,1)=history_j{1,inter-1}(i,1);
        end
      
        %确定搜索出来的值是否正确
        [~,I]=min(history_j{1,maxIter}(:));
        nets(1 ,:)=history_solution{1,maxIter}(I,:);
        % plot(fval);
        % xlabel('迭代次数')
        % ylabel('适应度函数值')
        

    end
    best_value=min(history_j{1,inter});
    history_best_value(inter)=best_value;
    the_best_value=min(history_best_value);

    
    
end

% %加入的模拟退火算法画图程序
% for kk=1:9
%     figure(3)
%     plot(fval{kk}(:),'Color','g');
%     xlabel('迭代次数');
%     ylabel('适应度值');
%     title('模拟退火算法历代最优值')
%     hold on
%     grid on
% end
% for kk=1:9
%     plot(fval{kk}(:));
%     hold on;
% end


t(1,q)=toc;%统计100次迭代平均时间
figure(1)
bar(t);
xlabel('Number of experiments')
ylabel('Time/s')
title('ALO algorithm runtime graph')
% xlabel('迭代次数')
% ylabel('时间/s')

%将ture值转换到新空间中
final_truth=(Z1.'*truth)';

tf=isequal(final_truth,nets);%统计迭代时搜索值的正确性
if tf
    truth_number=truth_number+1;
end
end

%绘图
figure(2)
plot(history_best_value);
xlabel('迭代次数')
ylabel('适应度函数值')
title('历史最优适应度值')
grid on % 添加网格


    

% %% PID搜索算法步长函数
% function step=PSA_levyFlight(beta)      
% sigma=(gamma(1+beta).*sin(pi*beta/2)./(gamma((1+beta)/2).*beta.*2.^((beta-1)/2))).^(1/beta); % 多了.^(1/beta)
% u=randn()*sigma;
% v=randn();
% step=u./abs(v).^(1/beta);
% end




%% 模拟退火算法1
%initial_solution: 初始解向量。
%cost_function: 成本函数，接受一个解向量作为输入，返回该解的成本。
%max_iterations: 最大迭代次数。
%initial_temperature: 初始温度。初始温度一般来说可以设置为问题空间的尺度范围，通常在较大的范围内进行尝试
%cooling_rate: 退火速率，控制温度的降低速度。通常设置为一个小于1的常数，一般在0.8到0.99之间，这个值也可以根据实际情况进行调整。

% function [best_solution] = simulated_annealing(a1,Q1,alos_position, maxInter, initial_temperature, cooling_rate)
%     % 初始化当前解和当前成本
%     current_solution = alos_position;
%     current_cost = (a1-current_solution)*inv(Q1)*(a1-current_solution)';
% 
%     % 初始化最佳解和最佳成本
%     best_solution = current_solution;
%     best_cost = current_cost;
% 
%     % 初始化温度
%     temperature = initial_temperature;
% 
%     % 迭代模拟退火算法
%     for iteration = 1:maxInter
%         % 生成一个随机解
%         new_solution = generate_neighbor(current_solution);
% 
%         % 计算新解的成本
%         new_cost = (a1-new_solution)*inv(Q1)*(a1-new_solution)';
% 
%         % 计算成本变化
%         cost_change = new_cost - current_cost;
% 
%         % 如果新解更优，则接受新解
%         if cost_change < 0 || exp(-cost_change / temperature) > rand()
%             current_solution = new_solution;
%             current_cost = new_cost;
% 
%             % 更新最佳解
%             if current_cost < best_cost
%                 best_solution = current_solution;
%                 best_cost = current_cost;
%             end
%         end
% 
%         % 降低温度
%         temperature = temperature * cooling_rate;
%     end
% end
% 
% function neighbor_solution = generate_neighbor(current_solution)
%     % 生成邻近解的简单示例：随机交换两个元素位置
%     n = numel(current_solution);
%     idx = randperm(n, 2);
%     neighbor_solution = current_solution;
%     neighbor_solution(idx(1)) = current_solution(idx(2));
%     neighbor_solution(idx(2)) = current_solution(idx(1));
% end


%% 模拟退火算法2
function [best_solution,fval] = simulated_annealing1(a1,Q1,alos_position,options)
% fun: 目标函数句柄
% x0: 初始解
% options: 选项结构体，包括以下字段：
%   T0: 初始温度
%   alpha: 降温速率
%   T_min: 终止温度
%   max_iter: 最大迭代次数
%   verbose: 是否打印输出
% 返回值：
%   x: 最优解
%   fval: 目标函数在最优解处的取值

% 设置默认选项
% default_options = struct('T0',100,'alpha',0.95,'T_min',1,...
%                          'max_iter',10,'verbose',false);
% if nargin < 3
%     options = default_options;
% else
%     options = merge_options(default_options,options);
% end

% 初始化参数
T = options.T0;
best_solution = alos_position;
fval = (a1-best_solution)*inv(Q1)*(a1-best_solution)';
iter = 0;
best_x = best_solution;
best_fval = fval;

% 开始迭代
while T > options.T_min && iter < options.max_iter
    % 产生新解
    new_x = best_solution + fix(randn(size(best_solution)));
    new_fval = (a1-new_x)*inv(Q1)*(a1-new_x)';
    delta_f = new_fval - fval;
    
    % 接受新解
    if delta_f < 0 || exp(-delta_f/T) > rand()
        best_solution = new_x;
        fval = new_fval;
        if fval < best_fval
            best_x = best_solution;
            best_fval = fval;
        end
    end
    
    % 降温
    T = options.alpha * T;
    
    % % 打印输出
    % if options.verbose
    %     fprintf('iter=%d, T=%g, fval=%g, best_fval=%g\n',iter,T,fval,best_fval);
    % end
    
    % % 更新迭代计数器
    % iter = iter + 1;
end

% 返回最优解和目标函数值
best_solution = best_x;
fval = best_fval;
end

% % 合并选项结构体
% function opt = merge_options(default_opt,opt)
% if isempty(opt)
%     opt = default_opt;
% else
%     fields = fieldnames(default_opt);
%     for i = 1:length(fields)
%         if ~isfield(opt,fields{i})
%             opt.(fields{i}) = default_opt.(fields{i});
%         end
%     end
% end
% end



%% Tent混沌映射
% %基于Tent映射的种群初始化
% function x_apply = Tent_int(dim,a,Lb,Ub)
% x(1,1)=(Lb(1,1)+rand()*(Ub(1,1)-Ub(1,1)))/Ub(1,1); %初始点
% 
% %根据Tent映射函数，生成后续初始种群
% for i=1:dim-1
%     if x(i)<a
%         x(i+1)=2*x(i);
%     elseif x(i)>=a
%         x(i+1)=2*(1-x(i)) ;
%     end
% end
% 
% %获取在粒子边界约束内的种群解
% x_apply=x.*Ub;
% 
% %粒子边界约束检查
% I=x_apply<Lb;
% x_apply(I)=Lb(I);
% U=x_apply>Ub;
% x_apply(U)=Ub(U);
% end


%% 烟花算法生成当前个体的特殊值
 % function RS=obtain_spec_alos(rand,alos_position,dim,lb,ub)
 %            pos = alos_position;
 %            for i = 1:dim
 %                if(rand<0.5)
 %                    pos(i) = pos(i)*normrnd(1,1);
 %                    % 超出范围则取模
 %                    if (pos(i)>ub || pos(i)<lb)
 %                        pos(i) = lb(i)+mod(pos(i),(ub(i)-lb(i)));
 % 
 %                    end
 %                end
 %            end
 %            RS(:,i) = pos(i);
 %        end




