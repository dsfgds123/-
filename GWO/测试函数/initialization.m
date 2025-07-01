% This function initialize the first population of search radiations (Agents)
% �˺������ڳ�ʼ���������䣨�����ĳ�ʼ��Ⱥ
function Positions=initialization(SearchAgents_no,dim,ub,lb)
    % ����߽�������������� ub ������
    Boundary_no= size(ub,2); % numnber of boundaries

    % If the boundaries of all variables are equal and user enter a single
    % number for both ub and lb
    % ������б����ı߽���ȣ����û�Ϊ���� ub ������ lb ��������һ����һ����ֵ
    if Boundary_no==1
        % ����һ����СΪ SearchAgents_no �С�dim �е�������󣬾���Ԫ�ط�Χ�� [0, 1] ֮��
        % Ȼ������� (ub - lb) ������ lb���õ��� [lb, ub] ��Χ�ڵĳ�ʼ��Ⱥλ��
        Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
    end

    % If each variable has a different lb and ub
    % ���ÿ���������в�ͬ������ lb ������ ub
    if Boundary_no>1
        % ����ÿ��ά��
        for i=1:dim
            % ��ȡ��ǰά�ȵ�����
            ub_i=ub(i);
            % ��ȡ��ǰά�ȵ�����
            lb_i=lb(i);
            % ����һ����СΪ SearchAgents_no �С�1 �е��������������Ԫ�ط�Χ�� [0, 1] ֮��
            % Ȼ������� (ub_i - lb_i) ������ lb_i���õ���ǰά���� [lb_i, ub_i] ��Χ�ڵĳ�ʼ��Ⱥλ��
            Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
        end
    end
end