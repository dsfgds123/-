% This function initialize the first population of search radiations (Agents)
% 此函数用于初始化搜索辐射（代理）的初始种群
function Positions=initialization(SearchAgents_no,dim,ub,lb)
    % 计算边界的数量，即上限 ub 的列数
    Boundary_no= size(ub,2); % numnber of boundaries

    % If the boundaries of all variables are equal and user enter a single
    % number for both ub and lb
    % 如果所有变量的边界相等，且用户为上限 ub 和下限 lb 都输入了一个单一的数值
    if Boundary_no==1
        % 生成一个大小为 SearchAgents_no 行、dim 列的随机矩阵，矩阵元素范围在 [0, 1] 之间
        % 然后将其乘以 (ub - lb) 并加上 lb，得到在 [lb, ub] 范围内的初始种群位置
        Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
    end

    % If each variable has a different lb and ub
    % 如果每个变量都有不同的下限 lb 和上限 ub
    if Boundary_no>1
        % 遍历每个维度
        for i=1:dim
            % 获取当前维度的上限
            ub_i=ub(i);
            % 获取当前维度的下限
            lb_i=lb(i);
            % 生成一个大小为 SearchAgents_no 行、1 列的随机向量，向量元素范围在 [0, 1] 之间
            % 然后将其乘以 (ub_i - lb_i) 并加上 lb_i，得到当前维度在 [lb_i, ub_i] 范围内的初始种群位置
            Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
        end
    end
end