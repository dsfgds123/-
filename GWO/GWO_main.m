% �����������������������ݣ��ر�����ͼ�δ��ڣ����رվ�����Ϣ
clear;clc;close all;warning off
% ����ǰ�ļ��м������ļ�����ӵ� MATLAB ������·���У��Ա���Ե������еĺ���
addpath(genpath(pwd));

%% ��ǰ10���Ż���������ѭ������
for i = 1:10
    % ������ i ת��Ϊ�ַ���
    F = sprintf('%d', i);
    % ���ɺ���������ʽΪ 'F' �������֣����� 'F1', 'F2' ��
    Function_name = ['F', F]; 
    % ���� Get_Functions_details_1 ��������ȡ��ǰ��ѡ�����������Ϣ
    % lb: �������½�
    % ub: �������Ͻ�
    % dim: ������ά��
    % fobj: Ŀ�꺯�����
    [lb, ub, dim, fobj] = Get_Functions_details_1(Function_name); 
    
    % ������Ⱥ���������������������
    nPop = 30; 
    % �����������������㷨���ڴﵽ�˵���������ֹͣ
    Max_iter = 1000; 

    %% �����㷨
    %% ��ʼ��һ���յ�Ԫ�����飬���ڱ����Ż����
    Optimal_results = {}; 
    % ��ʼ���������
    index = 1; 

    % ��ʼ��ʱ�����ڼ�¼�㷨������ʱ��
    tic
    % �����Ż��㷨������Ϊ "GWO"�������Ż��㷨��
    optimization_name = 'GWO'; 
    % ���� GWO ���������Ż���������Ⱥ���������������������½硢ά�Ⱥ�Ŀ�꺯�����
    % Best_score: �ҵ������ź���ֵ
    % Best_x: ��Ӧ�����ű���
    % cg_curve: �������ߣ���¼ÿ�ε��������ź���ֵ
    [Best_score, Best_x, cg_curve] = GWO(nPop, Max_iter, lb, ub, dim, fobj);
    % ���㷨���Ʊ��浽���������
    Optimal_results{1, index} = optimization_name; 
    % ���������߱��浽���������
    Optimal_results{2, index} = cg_curve; 
    % �����ź���ֵ���浽���������
    Optimal_results{3, index} = Best_score; 
    % �����ű������浽���������
    Optimal_results{4, index} = Best_x; 
    % ֹͣ��ʱ����������ʱ�䱣�浽���������
    Optimal_results{5, index} = toc; 
    % ������ 1��Ϊ��һ�������׼��
    index = index + 1; 

    % ��ʾ��Ϣ�����ʹ�� GWO �㷨Ϊ��ǰ�����ҵ�������Ŀ�꺯��ֵ
    display(['The best optimal value of the objective funciton found by ', [num2str(optimization_name)], ' for ', [num2str(Function_name)], '  is : ', num2str(Best_score)]);   

    %% �����������ߺ�Ŀ�꺯������άͼ
    % ����һ���µ�ͼ�δ��ڣ���������λ�úʹ�С
    figure; set(gcf, 'position', [300, 300, 800, 330])
    % ��ͼ�δ��ڻ���Ϊ 1 �� 2 �е���ͼ����ѡ���һ����ͼ
    subplot(1, 2, 1);
    % ���� func_plot_1 ���������Ƶ�ǰĿ�꺯������άͼ
    func_plot_1(Function_name)
    % ������ͼ�ı���Ϊ��ǰ������
    title(Function_name)
    % ���� x ���ǩ
    xlabel('x')
    % ���� y ���ǩ
    ylabel('y')
    % ���� z ���ǩ
    zlabel('z')

    % ѡ��ڶ�����ͼ
    subplot(1, 2, 2);
    % ������������е�ÿһ�����
    for i = 1:size(Optimal_results, 2)
        % ʹ�ð������������������ߣ������߿�Ϊ 2
        semilogy(Optimal_results{2, i}, 'Linewidth', 2)
        % ����ͼ�Σ��Ա���ͬһͼ�л��ƶ�������
        hold on
    end
    % ������ͼ�ı���Ϊ��ǰ������
    title(Function_name)
    % ���� x ���ǩΪ��������
    xlabel('Iteration');
    % ���� y ���ǩΪ��ǰ���������ŵ÷�
    ylabel(['Best score on ', num2str(Function_name)]);
    % ���������᷶Χ��ʹͼ�ν���
    axis tight
    % ��ʾ������
    grid on;
    % ��ʾͼ�α߿�
    box on
    % ��ʾͼ������עÿ�����߶�Ӧ���㷨����
    legend(Optimal_results{1, :})
end

% ��֮ǰ��ӵ�����·�����ļ��м������ļ��д�·�����Ƴ����ָ�ԭʼ·������
rmpath(genpath(pwd));