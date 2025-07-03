function J=target_function(a,Q,best_solution)

J=(a-best_solution)*inv(Q)*(a-best_solution)'; %适应度函数Y=min[(N-Nz)Q'(N-Nz)]

end 

% 
% function J=target_function(a,Q,best_solution)
% 
% J=log10((a-best_solution)*inv(Q)*(a-best_solution)'+1); %适应度函数Y=lg(J(N)+1);J(N)=min[(N-Nz)Q'(N-Nz)]
% 
% end