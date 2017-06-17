function [aOpt, xOpt, fEval, nIter, linprogTime] = FDP_LINPROG(n, N, l, m, s, tau, P, xBar, ipmMaxIter)

%% Function Argument Definitions
% n:          scalar number of tasks
% N:          scalar number of discrete time steps
% l:          scalar number of speed levels
% m:          scalar number of processors
% s:          column vector of speed levels [1xl]
% tau:        column vector of time values [(N+1)x1]
% P:          column vector of net power consumption values corresponding to each speed level [lx1]
% xBar:       column vector containing minimum execution time of tasks [nx1]
% ipmMaxIter: scalar number for maximum number of interior-point method iterations in linprog

%% [1] Optimisation Problem Definition
%% [1.1] Parameter Setup
P = P';   %converting to row vector
s = s';   %converting to row vector

% diffTau: row vector containing tau[k+1]-tau[k] values for k=0 to k=N-1 [1xN]
diffTau = tau(2:N+1)'-tau(1:N)';

len_a = N*l*n;
len_x = (N+1)*n;
Nn = len_x-n;
nl = n*l;

%% [1.2] Cost Function
% Note: the decision variable w is constructed as [a,x]'
% f1 and f2 are row vectors [1xNln] i.e. [1xlen_a]
% f is a column vector [(Nln+(N+1)n)x1] i.e. [(len_a+len_x)x1]
f1 = kron(diffTau,ones(1,nl));
f2 = kron(ones(1,N),kron(P,ones(1,n)));
f  = [(f1.*f2)';sparse(len_x,1)];

%% [1.3] Upper and Lower Bounds
% values of a are bounded by 0 and 1
% values of x are bounded by 0 and xBar
ub_a = ones(len_a,1);
ub_x = kron(ones((N+1),1),xBar);
UB   = [ub_a;ub_x];  %[1x(len_a+len_x)]
LB   = sparse(1,len_a+len_x);  %[1x(len_a+len_x)]

%% [1.4] Equality constraints
% Aeq0*w=beq0
% Set the remaining execution time for every task at arrival time to xBar
% This is a constraint on x only
Aeq0 = [sparse(n,len_a), speye(n,n), sparse(n, len_x-n)];
beq0 = xBar;

% Aeq1*w=beq1
% Set the remaining execution time for every task at its deadline to 0
% This is a constraint on x only
Aeq1 = [sparse(n,len_a), sparse(n, len_x-n), speye(n,n)];
beq1 = sparse(n,1);

% Aeq2*w=beq2
% Define the task scheduling dynamic
% This is a constraint on both a and x
Aeq2a = kron(diffTau.*speye(N), kron(speye(n), -s));
Aeq2x = spdiags([ones(Nn,1), -ones(Nn,1)], [0, n], Nn, len_x);
Aeq2  = [Aeq2a, Aeq2x];
beq2  = sparse(Nn,1);

Aeq=[Aeq0;Aeq1;Aeq2];
beq=[beq0;beq1;beq2];

%% [1.5] Inequality constraints
% A0*w<=b0
% Require that a only one task can only execute on one processor at a time
% This is a constraint on a only
A0a = kron(speye(N), kron(speye(n), ones(1,l)));
A0  = [A0a, sparse(Nn,len_x)];
b0  = ones(Nn,1);

% A1*w<=b1
% Require that the system's workload is not exceeded
% This is a constraint on a only
A1a = kron(speye(N), ones(1,nl));
A1 = [A1a, sparse(N, len_x)];
b1 = m.*ones(N,1);

A=[A0;A1];
b=[b0;b1];

%% [2] Solve using LINPROG
opts = optimoptions('linprog');
opts.Algorithm = 'interior-point';
opts.Display = 'off';
opts.MaxIterations = ipmMaxIter;

st=tic;
[w, fEval, exitFlag, output] = linprog(f,A,b,Aeq,beq,LB,UB,[],opts);
et=toc(st);
linprogTime = et;

if exitFlag~=1
    disp(exitFlag);
end

aOpt = w(1:len_a);
aOpt = reshape(aOpt, [nl,N]);
xOpt = w(len_a+1:end);
xOpt = reshape(xOpt, [n,N+1]);

nIter=output.iterations;
