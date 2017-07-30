function [solution, fEval, niter, timeperiter] = ak7213_linp(m, l, s, AP, IP, N, nj, tau, n, na, xd)

%% Function Argument Definitions
% m:   scalar number of processors
% l:   scalar number of speed levels
% s:   column vector of speed levels [1xl]
% AP:  column vector of active power datas corresponding to each speed level [lx1]
% IP:  scalar constant for idle power data
% N:   scalar number of discretization steps
% nj:  column vector containing number of deadlines at each step [nx1]
% tau: column vector of time values [(N+1)x1]
% n:   scalar number of tasks
% na:  column vector containing arrival time of tasks [nx1]
% xd:  column vector containing minimum execution time of tasks [nx1]

%% [1] Optimisation Problem Definition
%% [1.1] Parameter Setup
AP = AP'; %converting to row vector
s = s';   %converting to row vector

% diff_tau: row vector containing tau[k+1]-tau[k] values for k=0 to k=N-1 [1xN]
diff_tau = tau(2:N+1)'-tau(1:N)';

len_a = N*l*n;
len_x = (N+1)*n;
nN = len_x-n;
nl = n*l;
% nr: column vector containing remaining number of tasks at each time step, starting from k=1 [Nx1]
nr = zeros(N,1);
nr(1) = n;
for k = 1:N-1
    nr(k+1) = nr(k)-nj(k);
end

%% [1.2] Cost Function
% Note: the decision variable w is constructed as [a,x]'
% f1 and f2 are row vectors [1xNln] i.e. [1xlen_a]
% f is a column vector [(Nln+(N+1)n)x1] i.e. [(len_a+len_x)x1]
f1 = kron(diff_tau,ones(1,l*n));
f2 = kron(ones(1,N),(kron(AP(1:l),ones(1,n))-IP));
f  = [(f1.*f2)';zeros(len_x,1)];
clear f1 f2;

%% [1.3] Upper and Lower Bounds
% values of a are bounded by 0 and 1
% values of x are bounded by 0 and xd
ub_a = ones(len_a,1);
ub_x = kron(ones((N+1),1),xd);
UB   = [ub_a;ub_x];  %[1x(len_a+len_x)]
LB   = zeros(1,len_a+len_x);  %[1x(len_a+len_x)]
clear ub_a ub_x;

%% [1.4] Equality constraints
% Aeq0*w=beq0
% This equality sets the min execution time for every task at its corresponding arrival time
% This is a constraint on x
Aeq0 = sparse(n,len_a+len_x);
for i = 1:n
    Aeq0(i,len_a+(na(i)-1)*n+i) = 1;
end
beq0=xd;

% Aeq1*w=beq1
% This equality ensures that every task meets its deadline
% This is a constraint on x
Aeq1 = sparse(n,len_a+len_x);
for k = 1:N           %iterate over each step
    for j = 1:nj(k)   %iterate over number of deadlines at each step
    %k*n cycles through the steps
    %n-nr(k) cycles through the task set that must be complete at each step
    %j cycles through each task in the due task set
    Aeq1(n-nr(k)+ j , len_a+ k*n+ n-nr(k)+ j) = 1;
    end
end
beq1=zeros(n,1);

% Aeq2*w=beq2
% This equality defines the task scheduling dynamic
% This is a constraint on both a and x
Aeq2 = [];
for k = 1:N           %iterate over each step
    %kron(step*s, speye(n)) contains the step terms we are interested in
    %the two surrounding sparse matrices allow us to point at the correct a variables
    step = tau(k+1)-tau(k);
    Aeq2 = [Aeq2; [sparse(n, nl*(k-1)), kron(step*s, speye(n)), sparse(n, nl*(N-k)+ len_x)] ];
end
Aeq2 = Aeq2+ [sparse(nN, len_a+n), speye(nN)]+ [sparse(nN, len_a), -speye(nN), sparse(nN,n)];
beq2=zeros(nN,1);

Aeq=[Aeq0;Aeq1;Aeq2];
beq=[beq0;beq1;beq2];
clear Aeq0 Aeq1 Aeq2 beq0 beq1 beq2;

%% [1.5] Inequality constraints
% A0*w<=b0
% This inequality ensures that a single task can only execute on one processor at a time
% This is a constraint on a
A0 = [kron(speye(N), kron(ones(1,l), speye(n))), sparse(nN,len_x)];
b0 = ones(nN,1);

% A1*w<=b1
% This inequality ensures that the system's workload is not exceeded
% This is a constraint on a
A1=[kron(speye(N), ones(1,nl)), sparse(N,len_x)];
b1=m.*ones(N,1);

A=[A0;A1];
b=[b0;b1];

clear A0 A1 b0 b1;

%% [2] linprog
% Output: [solution, fEval, niter, timeperiter]

opts = optimoptions('linprog','display','off','Algorithm','interior-point');

st=tic;
[solution, fEval, exitFlag, output] = linprog(f,A,b,Aeq,beq,LB,UB,[],opts);
et=toc(st);

niter=output.iterations;
timeperiter = et/niter;

if exitFlag~=1
disp(exitFlag);
end

%% Note:
% If we have 4 tasks, we set n = 4;
% The minimum execution time is set as xd = [4; 4; 2; 2];
% The arrival time is set as na = [1; 1; 1; 2];
% That means the first three tasks arrive at k = 1
% and the last one task arrives at k = 2
% The deadlines are set as nj = [2; 2;];
% That means the first two tasks have the deadlines at k = 1;
% and the last two tasks have the deadlines at k = 2;
