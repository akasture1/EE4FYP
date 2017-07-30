n = 10; N = 10; l = 10; m = 1000;


s = (1/l:1/l:1)';
step = 5;
tau = (0:step:step*N)';
xBar = step*rand(n,1);

% Simplified Dynamic Power Consumption Model: AP(s) = alpha * s^beta
% alpha > 0 ; beta >= 1
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);

ipmTol = 1e-4;
solverTol = 1e-6;
ipmMaxIter = 200;
solverMaxIter = 20000;
ipmIter= 1;