rng(2)
close all
clc
clear
profile on

% Problem Size
n = 80;
N = 40;
l = 10;
m = 100;

s = (1/l:1/l:1)';
step = 5;
tau = (0:step:step*N)';

% AP = 1000*((1/l:1/l:1).^2)';
% IP = 40;

AP = 10*((1/l:1/l:1).^2)';
IP = -mean(AP);

na = ones(n,1); 
nj = n/N*ones(n,1);
xBar = step*rand(n,1)+1;
ipmTol = 1e-4;

[aOpt, xOpt, fEval, niter, timeperiter] = ak7213_linp(m, l, s, AP, IP, N, nj, tau, n, na, xBar);
%[cost, aOpt, xOpt, niter] = siquan_LPDVFS(m, l, s, AP, IP, N, nj, tau, n, na, xBar, ipmTol);

figure
plot(0:1:N,xOpt', 'LineWidth', 2);
%plot(0:1:N,[xBar, xOpt]', 'LineWidth', 2);
title('Same-Arrival-Different-Deadline');
xlabel('Time Step [k]');
ylabel('Remaining Time (s)');
grid on
grid minor


profile report
profile off
