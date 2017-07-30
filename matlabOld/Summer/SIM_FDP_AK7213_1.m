%% SIM_FDP_AK7213_1
% Fixed Deadline Problem
% Simulation using Custom IPM
% Increasing number of Tasks [n]

% Author: Anand Kasture (ak7213@ic.ac.uk)
% Date: June 2017

rng(2)
clear

%% [Sim #1] Increasing number of Tasks (n)
N = 10; l = 10; m = 1000;

s = (1/l:1/l:1)';
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);

step = 5;
tau = (0:step:step*N)';

ipmTol = 1e-4;
ipmMaxIter = 40;
solver = 3;
solverTol = 1e-6;

vals = [200:100:1000];
Obs = length(vals);
nIters = zeros(Obs,1);
tSolv = zeros(Obs,1);
j = 1;

fprintf('[SIM_FDP_AK7213][Sim #1] Increasing number of Tasks [n]\n');
for n = vals
    xBar = step*rand(n,1)+1;
    ts = tic;
    [aOpt, xOpt, nIter] = FDP_AK7213_V1(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol);
    %[aOpt, xOpt, nIter] = FDP_AK7213_V2(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol);
    te = toc(ts);
    tSolv(j) = te;
    nIters(j) = nIter;
    j = j+1;
end

bigGrid = vals';
fineGrid = (vals(1):vals(end))';
eqn = fit(bigGrid, tSolv./nIters,'power2');
bestFitLine = eqn.a * fineGrid.^eqn.b + eqn.c;


figure
plot(bigGrid, tSolv, '-o','LineWidth', 2, 'MarkerSize', 10)
title('Function Duration vs Number of Tasks [FDP-AK7213-V1]');
xlabel('Number of Tasks');
ylabel('Time (s)');
grid on

figure
plot(bigGrid, nIters, '-o','LineWidth', 2, 'MarkerSize', 10)
title('Number of Iterations vs Number of Tasks [FDP-AK7213-V1]');
xlabel('Number of Tasks');
ylabel('Number of Iterations');
grid on

figure
loglog(bigGrid, tSolv./nIters, 'x', 'LineWidth', 2, 'MarkerSize', 15)
hold on
loglog(fineGrid, bestFitLine, 'LineWidth', 2)
xlim([fineGrid(1) fineGrid(end)])
title('Time Per Iteration vs Number of Tasks [FDP-AK7213-V1]');
xlabel('Number of Tasks');
ylabel('Time Per Iteration (s)');
grid on

fprintf('Line of Best Fit: (%3.6f)x^(%3.6f) + %3.6f\n', eqn.a, eqn.b, eqn.c);
save('SIM_FDP_AK7213_SIM1')
