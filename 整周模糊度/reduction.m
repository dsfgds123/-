function [Z,L,D1]=reduction (L_,D_)%降相关
%进行Z变换
n=length(L_);
L=L_;
D=diag(D_);
j=n-2+1;%7
k=n-2+1;%7
Z=eye(n);%单位阵
while j>=1
    if j<=k
        for i=j+1:n
            [Z,L]=gauss(n,L,Z,i,j);
        end
    end
    
    del=D(j)+L(j+1,j)*L(j+1,j)*D(j+1);%del为D（j+1）交换后的值
    
    if (del+1e-6)<D(j+1)%将交换后的值与原来的值做比较
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