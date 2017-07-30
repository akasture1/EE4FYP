function [cost, out, niter, timeperiter] = linp(m, l, s, AP, IP, N, nj, tau, n, na, xd)
%%
% m: number of processors
% l: number of speed levels
% s: vector of speed levels
% AP: vector of active power datas
% IP: idle power data
% N: number of discretization steps
% nj: a vector containing number of deadlines at each step
% tau: time grid
% n: number of tasks
% na: arrival time of tasks
% xd: minimum execution time of tasks
%% 
% Note:
% If we have 4 tasks, we set n = 4;
% The minimum execution time is set as xd = [4; 4; 2; 2];
% The arrival time is set as na = [1; 1; 1; 2];
% That means the first three tasks arrive at k = 1 
% and the last one task arrives at k = 2
% The deadlines are set as nj = [2; 2;];
% That means the first two tasks have the deadlines at k = 1;
% and the last two tasks have the deadlines at k = 2;
%% The first part of this program is to generate the necessary parameters and matrices
UX=N+1;
x=xd;
AP = AP';
S = s';
nr = zeros(N,1);
nr(1) = n;
for k = 1:N-1
    nr(k+1) = nr(k)-nj(k);
end

%% Cost Function
f1=kron(tau(2:N+1)'-tau(1:N)',ones(1,n*l));
f2=kron(AP(1:l),ones(1,n))-IP;
f2=kron(ones(1,N),f2);
f=[(f1.*f2)';zeros(n*UX,1)];
clear f1 f2;
% min f'*[a;x]
% a index = I * Q * U
% x index = I * UX

%% UB and LB 
LB=zeros(1,n*l*N+n*UX);
VU=ones(UX,1);
ub=kron(VU,x(1:n));% not essential
UB=[ones(n*l*N,1);ub];
clear VU ub;

%% Equal constraints
Aeq0 = sparse(n,n*l*N+UX*n);
for i = 1:n
    Aeq0(i,n*l*N+(na(i)-1)*n+i) = 1;
end

Aeq1=sparse(n,n*l*N+UX*n);
for k = 1:N
    for j = 1:nj(k)
    Aeq1(n-nr(k)+j,n*l*N+k*n+n-nr(k)+j)=1;
    end
end
beq1=x;
beq2=zeros(n,1);

Aeq3 = [];
for k = 1:N
    Aeq3 = [Aeq3;[sparse(n,n*l*(k-1)),kron((tau(k+1)-tau(k))*S,speye(n)),sparse(n,n*l*(N-k)+(N+1)*n)]];
end
Aeq3 = Aeq3+[sparse(n*N,n*l*N+n),speye(n*N)]+[sparse(n*N,n*l*N),-speye(n*N),sparse(n*N,n)];

beq3=zeros(n*N,1);

Aeq=[Aeq0;Aeq1;Aeq3];
beq=[beq1;beq2;beq3];
clear Aeq1 Aeq3 beq1 beq2 beq3;

%% Inequal constraint
A1=[kron(eye(N),kron(ones(1,l),speye(n))),sparse(n*N,n*UX)];
b1=ones(n*N,1);

A2=[kron(speye(N),ones(1,n*l)),sparse(N,n*UX)];
b2=m.*ones(N,1);
A=[A1;A2];
b=[b1;b2];
clear A1 b1 b2;

% The second part of this program is to solve the LP
%% Solve 
opts = optimoptions('linprog','display','off','Algorithm','interior-point');

st=tic;
[cost,out,FLAG,OUT]=linprog(f,A,b,Aeq,beq,LB,UB,[],opts);
et=toc(st);
niter=OUT.iterations;
timeperiter = et/niter;
if FLAG~=1
disp(FLAG);
end
