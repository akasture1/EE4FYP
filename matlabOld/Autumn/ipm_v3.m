close all
clear all

%% Parameter Setup

% m:   scalar number of processors
% l:   scalar number of speed levels
% s:   column vector of speed levels [1xl]
% AP:  column vector of active power datas corresponding to each speed level [lx1]
% IP:  scalar constant for idle power data
% n:   scalar number of tasks
% N:   scalar number of discretization steps
% tau: column vector of time values [(N+1)x1]
% xd:  column vector containing minimum execution time of tasks [nx1]
% na:  column vector containing arrival time of tasks [nx1]
% nj:  column vector containing number of deadlines at each step [nx1]

% Simplified Dynamic Power Consumption Model: AP(s) = alpha * s^beta
% alpha > 0 ; beta >= 1

m = 10;
l = 4; 
s = (1/l:1/l:1)';

AP = 1000*((1/l:1/l:1).^2)';
IP = 40;
P = AP-IP;

n = 4;
N = 8;
step = 5;
tau = (0:step:step*N)';
xd = step*rand(n,1);


% All tasks have the same arrival time (k=0) and same deadline (k=N)
na = ones(n,1); 
nj = N*ones(n,1); 

%% Mehrotra's Predictor-Corrector IPM
tol = 1e-5;
iter=0;
nl = n*l;
nd = nl + (n+1);

%----------------------------------------------------------------%
% Choose feasible starting point
% using +ve lagrange multipliers (p, lambda) and slack variable (t)
x = [xd, zeros(n,N)];
a = zeros(nl,N);
lambda = ones(nd,N);    
p = ones(n,N);
t = ones(nd,N);  
beta = ones(n,1);

%----------------------------------------------------------------%
% Precalcualte matrices/vectors that remain constant over all iterates
d = [ones(n,1); m  ; zeros(nl,1)];

D1 = kron(speye(n), ones(1,l));
D2 = ones(1,nl);
D3 = -1*speye(nl);
D = [D1;D2;D3];
%dim = size(D);
%assert(dim(1)==nd && dim(2)==n);
clear D1 D2 D3

B = kron(-1*step*speye(n), s');
%dim = size(B);
%assert(dim(1)==n && dim(2)==nl);

T = spdiags(t(:,1),0,nd,nd);
invT = spdiags(t(:,1).^-1,0,nd,nd);
%dim = size(T);
%assert(dim(1)==nd && dim(2)==nd);

Lambda = spdiags(lambda(:,1),0,nd,nd);
invLambda = spdiags(lambda(:,1).^-1,0,nd,nd);
%dim = size(Lambda);
%assert(dim(1)==nd && dim(2)==nd);

% Block Elimination
E = invLambda*T;
invE = invT * Lambda;
R = D'*invE*D;
%----------------------------------------------------------------%
% Set up Predictor RHS
rx = zeros(n,N+1);
ra = zeros(nl,N);
rlambda = zeros(nd,N);    
rp = zeros(n,N);
rt = zeros(nd,N);
rbeta = zeros(n,1);

rlambda_hat = zeros(nd,N);
ra_hat = zeros(nl,N);

for k = 1:N
    ra(:,k) = step * kron(ones(n,1), P) + B'*p(:,k) + D'*lambda(:,k);
    rlambda(:,k) = D * a(:,k) - d + t(:,k);
    rp(:,k) = x(:,k) + B * a(:,k) - x(:, k+1);
    rt(:,k) = Lambda * T * ones(nd,1);
    
    rlambda_hat(:,k) = rlambda(:,k) + invLambda*rt(:,k);
    ra_hat(:,k) = D' * invE * rlambda_hat(:,k);
end

rx(:,1) = p(:,1);
for k = 2:N
    rx(:,k) = p(:,k) - p(:,k-1);
end
rx(:,N+1) = beta - p(:,N);
rbeta = x(:,N+1);

cascadedRs = vertcat(rx(:,1:N), ra_hat, rp);
RHS = [cascadedRs(:); rx(:,N+1); rbeta];

%----------------------------------------------------------------%
% Set up Predictor LHS
dim = [n, nl, n];
pos = zeros(3,2);   %start and end points for each sub-block e.g. B, D etc.

pos(1,1) = 1;
pos(1,2) = dim(1);
for k = 2:size(dim,2)
    pos(k,1) = sum(dim(1:k-1)) + 1;
    pos(k,2) = pos(k,1) + dim(k) - 1;
end

uBlock = sparse(sum(dim),sum(dim));

%(1,3) I
I = speye(n);
x=1;y=3;
uBlock(pos(x,1):pos(x,2), pos(y,1):pos(y,2)) = I;    

%(2,2) R
x=2;y=2;
uBlock(pos(x,1):pos(x,2), pos(y,1):pos(y,2)) = R;    

%(2,3) B'
x=2;y=3;
uBlock(pos(x,1):pos(x,2), pos(y,1):pos(y,2)) = B';    

%(3,1) I
I = speye(n);
x=3;y=1;
uBlock(pos(x,1):pos(x,2), pos(y,1):pos(y,2)) = I;  

%(3,2) B
x=3;y=2;
uBlock(pos(x,1):pos(x,2), pos(y,1):pos(y,2)) = B; 

% Compute banded matrix, and resize to accommodate last few blocks
LHS = kron(speye(N),uBlock);
[h,w] = size(LHS);
[i,j,s] = find(LHS);

% Add the 2 missing -I matrices in position (4,3) and (3,4) that we skipped
blockdim = sum(dim);

for k = 1:N-1
    p1 = (k*blockdim +1 :1: k*blockdim + n)';
    p2 = ((k-1)*blockdim + pos(3,1): 1 :(k-1)*blockdim +pos(3,2))';
    i = [i; p1; p2];
    j = [j; p2; p1];
    s = [s; -1*ones(2*n,1)];
end

k = N;
p1 = (k*blockdim +1 :1: k*blockdim + 2*n)';
p2 = ((k-1)*blockdim + pos(3,1): 1 :(k-1)*blockdim + pos(3,2) + n)';
i = [i; p1; p2];
j = [j; p2; p1];
s = [s; repmat([-1*ones(n,1); ones(n,1)],2,1)];

LHS = sparse(i,j,s,(h+2*n), (w+2*n));
%----------------------------------------------------------------%
% Solve for Predictor Step

dx = zeros(n,N+1);
da = zeros(nl,N);
dlambda = zeros(nd,N);    
dp = zeros(n,N);
dt = zeros(nd,N);
dbeta = zeros(n,1);

%----------------------------------------------------------------%


D = LHS \ RHS;
  
%dgap = lambda(:)'*t(:)/nd;
% while dgap > tol
%     
%     
%     
% dgap = calcDualityGap(lambda, t, nd, N);
% iter = iter+1;    
% end  

%% Functions

% Calculate the duality gap
function [dgap] = calcDualityGap(lambda, t ,nd, N)
    sum = 0;
    for k = 1:N
        sum = sum + lambda(:,k)'*t(:,k);
    end
    dgap = sum/nd;
end






