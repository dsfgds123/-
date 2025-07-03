% 蚁狮初始化函数
function X = initialization_ALO(pop, ub, lb, dim)
%input
%pop 种群数量
%ub 每个纬度变量上边界
%lb 每个纬度变量下边界
%dim 问题纬度
%ouput
%X 输出种群，size为[pop, dim]
    for i = 1:pop
        for j = 1:dim
            X(i, j) = (ub(j) - lb(j)) * rand() + lb(j);
        end
    end
end