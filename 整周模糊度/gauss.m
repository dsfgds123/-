function [Zi,L]=gauss(n,L,Zi,i,j)  %GPS���������ݴ���P175
%���и�˹�任
mu=round(L(i,j));%�ж��Ƿ�С�ڵ���0.5
if(mu~=0)
    for k=i:n
        L(k,j)= L(k,j)-mu*L(k,i);
    end
    for k=1:n
        Zi(k,j)=Zi(k,j)-mu*Zi(k,i);%Z��λ�任����ZiΪ���������ı仯����
    end
end
end