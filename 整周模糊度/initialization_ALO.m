% ��ʨ��ʼ������
function X = initialization_ALO(pop, ub, lb, dim)
%input
%pop ��Ⱥ����
%ub ÿ��γ�ȱ����ϱ߽�
%lb ÿ��γ�ȱ����±߽�
%dim ����γ��
%ouput
%X �����Ⱥ��sizeΪ[pop, dim]
    for i = 1:pop
        for j = 1:dim
            X(i, j) = (ub(j) - lb(j)) * rand() + lb(j);
        end
    end
end