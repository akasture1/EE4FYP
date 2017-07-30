rng(2)
clc
clear
close all

n = 100;
N = 100;
l = 10;
m = 1000;

s = (1/l:1/l:1)';
step = 5;
tau = (0:step:step*N)';

% Simplified Dynamic Power Consumption Model: AP(s) = alpha * s^beta
% alpha > 0 ; beta >= 1
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);

ipmTol = 5e-4;
solverTol = 1e-6;
ipmMaxIter = 60;
ipmIter= 1;

tPerIter = zeros(10,1);
j = 1;
for n = 100:100:2000
    xBar = step*rand(n,1)+1;
    ts = tic;
    [a, x, ipmIter] = ak7213IPMSolver (n,N,l,m,s,P,xBar,ipmTol,solverTol,ipmMaxIter);
    te = toc(ts);
    tPerIter(j) = te/ipmIter;
end

