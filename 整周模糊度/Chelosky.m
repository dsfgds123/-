function [L,D]=Chelosky(Q)
n=length(Q);%获得有多少个模糊度
L=zeros(n);%提前定义L矩阵
D=zeros(n);%提前定义D矩阵
D(n,n)=Q(n,n);%确定D矩阵最后一个元素
for i=1:1:n%确定L矩阵最后一行元素
    L(n,i)=Q(n,i)/D(n,n);
end
for i=n-1:-1:1%由右向左，由下向上获取Q矩阵与L矩阵中的所有元素i为行标
    %首先确定D矩阵中的元素
    A=0;
    for j=i+1:1:n
        A=A+D(j,j)*L(j,i)*L(j,i);
    end
    D(i,i)=Q(i,i)-A;
    %其次确定该行中的L矩阵元素
    for j=i:-1:1%由右向左获取L矩阵中的元素
        B=0;
        for k=j+1:1:n
            B=B+L(k,i)*D(k,k)*L(k,j);
        end
        L(i,j)=(Q(i,j)-B)/D(i,i);
    end
end
end