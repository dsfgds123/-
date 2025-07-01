function [Z,L,D]=perm(n,L,D,j,del,Z)
%进行置换   且Z为列交换
eta=D(j)/del;
lam=D(j+1)*L(j+1,j)/del;
D(j)=eta*D(j+1);
D(j+1)=del;
for k=1:j-1
    a0=L(j,k);
    a1=L(j+1,k);
    L(j,k)= -L(j+1,j)*a0+a1;
    L(j+1,k)=eta*a0+lam*a1;
end
L(j+1,j)=lam;
for k=j+2:n
    g=L(k,j);
    L(k,j)=L(k,j+1);
    L(k,j+1)=g;
end
for k=1:n    %将Z中列相互置换
    h=Z(k,j);
    Z(k,j)=Z(k,j+1);
    Z(k,j+1)=h;
end
end