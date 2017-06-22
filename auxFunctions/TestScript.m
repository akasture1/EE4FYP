% SIM_FDP_AK7213_1
% SIM_FDP_AK7213_3
% SIM_FDP_AK7213_2

rng(2)
clear
close all
clc

%% [Sim #1] Increasing number of Tasks (n)
n = 200; N = 20; l = 10; m = 1000;

s = (1/l:1/l:1)';
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);

step = 5;
tau = (0:step:step*N)';

ipmTol = 1e-4;
ipmMaxIter = 100;
solver = 1;
solverTol = 1e-3;
xBar = step*rand(n,1);

profile on
%[aOpt, xOpt, nIter] = FDP_AK7213_V1(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol);
[aOpt, xOpt, nIter] = DDP_AK7213_V4_CGS(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol);
profile report
profile off

plot((0:1:N)',[xBar xOpt]');
grid on
grid minor