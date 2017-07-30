% SIM_FDP_AK7213_1
% SIM_FDP_AK7213_3
% SIM_FDP_AK7213_2

rng(2)
clear
close all
clc

%% [Sim #1] Increasing number of Tasks (n)
n = 400; N = 50; l = 10; m = 1000;

s = (1/l:1/l:1)';
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);

step = 5;
tau = (0:step:step*N)';

ipmTol = 1e-6;
ipmMaxIter = 40;
solver = 3;
solverTol = 1e-3;
solverMaxIter = N+1;

xBar = step*rand(n,1);

profile on
%[aOpt, xOpt, nIter] = FDP_AK7213_V1(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol);
[aOpt, xOpt, nIter, cgsIters] = DDP_AK7213_V4_CGS(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol, solverMaxIter);
profile report
profile off

figure
plot((0:1:N)',[xBar xOpt]');
grid on
grid minor

figure
plot(cgsIters, 'LineWidth', 2);
grid on;