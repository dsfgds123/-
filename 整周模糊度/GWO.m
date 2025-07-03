%___________________________________________________________________%
%  Grey Wolf Optimizer (GWO) source codes version 1.0               %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili, S. M. Mirjalili, A. Lewis             %
%               Grey Wolf Optimizer, Advances in Engineering        %
%               Software , in press,                                %
%               DOI: 10.1016/j.advengsoft.2013.12.007               %
%                                                                   %
%___________________________________________________________________%

% Grey Wolf Optimizer
% �ú���ʵ���˻����Ż��㷨���������Ŀ�꺯��������ֵ
% �������:
% SearchAgents_no: �����������ǣ�������
% Max_iter: ����������
% lb: �������½�
% ub: �������Ͻ�
% dim: ������ά��
% fobj: Ŀ�꺯�����
% �������:
% Alpha_score: ���Ž��Ŀ�꺯��ֵ
% Alpha_pos: ���Ž��λ��
% Convergence_curve: �������ߣ���¼ÿ�ε���������Ŀ�꺯��ֵ
function [Alpha_score,Alpha_pos,Convergence_curve]=GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj)

% initialize alpha, beta, and delta_pos
% ��ʼ�� Alpha �ǵ�λ�ú���Ӧ��ֵ
% Alpha �Ǵ���ǰ�ҵ������Ž�
Alpha_pos=zeros(1,dim);
Alpha_score=inf; % ��ʼ��Ϊ�������Ϊ������⣬���Ϊ -inf
% ��ʼ�� Beta �ǵ�λ�ú���Ӧ��ֵ
% Beta �Ǵ���ǰ�ҵ��Ĵ��Ž�
Beta_pos=zeros(1,dim);
Beta_score=inf; % ��ʼ��Ϊ�������Ϊ������⣬���Ϊ -inf
% ��ʼ�� Delta �ǵ�λ�ú���Ӧ��ֵ
% Delta �Ǵ���ǰ�ҵ��ĵ����Ž�
Delta_pos=zeros(1,dim);
Delta_score=inf; % ��ʼ��Ϊ�������Ϊ������⣬���Ϊ -inf

%Initialize the positions of search agents
% ���� initialization ������ʼ�����������������ǣ���λ��
Positions=initialization(SearchAgents_no,dim,ub,lb);

% ��ʼ���������ߣ����ڼ�¼ÿ�ε���������Ŀ�꺯��ֵ
Convergence_curve=zeros(1,Max_iter);

% ��ʼ������������
l=0; 

% Main loop
% ��ѭ����ֱ���ﵽ����������
while l<Max_iter
    % ����ÿ�������������ǣ�
    for i=1:size(Positions,1)  
        % Return back the search agents that go beyond the boundaries of the search space
        % ������������Ƿ񳬳������ռ���Ͻ�
        Flag4ub=Positions(i,:)>ub;
        % ������������Ƿ񳬳������ռ���½�
        Flag4lb=Positions(i,:)<lb;
        % �������߽�������������ص������ռ���
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % Calculate objective function for each search agent
        % ���㵱ǰ���������Ŀ�꺯��ֵ
        fitness=fobj(Positions(i,:));
        
        % Update Alpha, Beta, and Delta
        % �����ǰ�����������Ӧ��ֵ���� Alpha �ǵ���Ӧ��ֵ
        if fitness<Alpha_score 
            % ���� Alpha �ǵ���Ӧ��ֵ
            Alpha_score=fitness; 
            % ���� Alpha �ǵ�λ��
            Alpha_pos=Positions(i,:);
        end
        
        % �����ǰ�����������Ӧ��ֵ���� Alpha �Ǻ� Beta ��֮��
        if fitness>Alpha_score && fitness<Beta_score 
            % ���� Beta �ǵ���Ӧ��ֵ
            Beta_score=fitness; 
            % ���� Beta �ǵ�λ��
            Beta_pos=Positions(i,:);
        end
        
        % �����ǰ�����������Ӧ��ֵ���� Beta �Ǻ� Delta ��֮��
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
            % ���� Delta �ǵ���Ӧ��ֵ
            Delta_score=fitness; 
            % ���� Delta �ǵ�λ��
            Delta_pos=Positions(i,:);
        end
    end
    
    % ������Ʋ��� a����������������Դ� 2 ��С�� 0
    a=2-l*((2)/Max_iter); 

    % Update the Position of search agents including omegas
    % ����ÿ�������������ǣ�
    for i=1:size(Positions,1)
        % ����ÿ������ά��
        for j=1:size(Positions,2)     
            % ����һ�� [0, 1] ֮��������
            r1=rand(); 
            % ����һ�� [0, 1] ֮��������
            r2=rand(); 
            
            % ���ݹ�ʽ (3.3) ���� A1
            A1=2*a*r1-a; 
            % ���ݹ�ʽ (3.4) ���� C1
            C1=2*r2; 
            
            % ���ݹ�ʽ (3.5) ���㵱ǰ�������� Alpha �ǵľ���
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); 
            % ���ݹ�ʽ (3.6) ���㵱ǰ���������� Alpha ���ƶ���λ��
            X1=Alpha_pos(j)-A1*D_alpha; 
            
            % ����һ�� [0, 1] ֮��������
            r1=rand();
            % ����һ�� [0, 1] ֮��������
            r2=rand();
            
            % ���ݹ�ʽ (3.3) ���� A2
            A2=2*a*r1-a; 
            % ���ݹ�ʽ (3.4) ���� C2
            C2=2*r2; 
            
            % ���ݹ�ʽ (3.5) ���㵱ǰ�������� Beta �ǵľ���
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); 
            % ���ݹ�ʽ (3.6) ���㵱ǰ���������� Beta ���ƶ���λ��
            X2=Beta_pos(j)-A2*D_beta; 
            
            % ����һ�� [0, 1] ֮��������
            r1=rand();
            % ����һ�� [0, 1] ֮��������
            r2=rand(); 
            
            % ���ݹ�ʽ (3.3) ���� A3
            A3=2*a*r1-a; 
            % ���ݹ�ʽ (3.4) ���� C3
            C3=2*r2; 
            
            % ���ݹ�ʽ (3.5) ���㵱ǰ�������� Delta �ǵľ���
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); 
            % ���ݹ�ʽ (3.5) ���㵱ǰ���������� Delta ���ƶ���λ��
            X3=Delta_pos(j)-A3*D_delta; 
            
            % ���ݹ�ʽ (3.7) ���µ�ǰ���������λ��
            Positions(i,j)=(X1+X2+X3)/3;
        end
    end
    % ������������ 1
    l=l+1;    
    % ��¼��ǰ����������Ŀ�꺯��ֵ
    Convergence_curve(l)=Alpha_score;
end