function [Zi,L]=gauss(n,L,Zi,i,j)  %GPS测量与数据处理P175
%进行高斯变换
mu=round(L(i,j));%判断是否小于等于0.5
if(mu~=0)
    for k=i:n
        L(k,j)= L(k,j)-mu*L(k,i);
    end
    for k=1:n
        Zi(k,j)=Zi(k,j)-mu*Zi(k,i);%Z单位变换矩阵，Zi为经过排序后的变化矩阵
    end
end
end