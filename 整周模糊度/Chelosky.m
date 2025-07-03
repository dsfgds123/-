function [L,D]=Chelosky(Q)
n=length(Q);%����ж��ٸ�ģ����
L=zeros(n);%��ǰ����L����
D=zeros(n);%��ǰ����D����
D(n,n)=Q(n,n);%ȷ��D�������һ��Ԫ��
for i=1:1:n%ȷ��L�������һ��Ԫ��
    L(n,i)=Q(n,i)/D(n,n);
end
for i=n-1:-1:1%���������������ϻ�ȡQ������L�����е�����Ԫ��iΪ�б�
    %����ȷ��D�����е�Ԫ��
    A=0;
    for j=i+1:1:n
        A=A+D(j,j)*L(j,i)*L(j,i);
    end
    D(i,i)=Q(i,i)-A;
    %���ȷ�������е�L����Ԫ��
    for j=i:-1:1%���������ȡL�����е�Ԫ��
        B=0;
        for k=j+1:1:n
            B=B+L(k,i)*D(k,k)*L(k,j);
        end
        L(i,j)=(Q(i,j)-B)/D(i,i);
    end
end
end