function [Z,L,D1]=reduction (L_,D_)%�����
%����Z�任
n=length(L_);
L=L_;
D=diag(D_);
j=n-2+1;%7
k=n-2+1;%7
Z=eye(n);%��λ��
while j>=1
    if j<=k
        for i=j+1:n
            [Z,L]=gauss(n,L,Z,i,j);
        end
    end
    
    del=D(j)+L(j+1,j)*L(j+1,j)*D(j+1);%delΪD��j+1���������ֵ
    
    if (del+1e-6)<D(j+1)%���������ֵ��ԭ����ֵ���Ƚ�
        [Z,L,D]=perm(n,L,D,j,del,Z);
        k=j;
        j=n-1;
    else
        j=j-1;
    end
end
for i=1:1:length(D)
    D1(i,i)=D(i);
end
end