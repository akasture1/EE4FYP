%% SIM_FDP_AK7213_2
% Fixed Deadline Problem
% Simulation using Custom IPM
% Increasing number of Time Steps [N]

% Author: Anand Kasture (ak7213@ic.ac.uk)
% Date: June 2017

rng(2)
clear

%% [Sim #2] Increasing number of Steps (N)
n = 10; l = 10; m = 100;

s = (1/l:1/l:1)';
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);

step = 5;
xBar = step*rand(n,1)+1;

ipmTol = 1e-4;
ipmMaxIter = 40;
solver = 1;
solverTol = 1e-6;

vals = [100:100:1000]
Obs = length(vals);
nIters = zeros(Obs,1);
tSolv = zeros(Obs,1);
j = 1;

fprintf('[SIM_FDP_AK7213][Sim #2] Increasing number of Time Steps [N]\n');
for N = vals
    tau = (0:step:step*N)';
    fprintf('starting  [n=%d; N=%d; l=%d; m=%d]\n',n,N,l,m);
    ts = tic;
    [aOpt, xOpt, nIter] = FDP_AK7213_V1(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol);
    %[aOpt, xOpt, nIter] = FDP_AK7213_V2(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol);
    te = toc(ts);
    fprintf('completed [n=%d; N=%d; l=%d; m=%d]\n',n,N,l,m);
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
title('Function Duration vs Number of Steps [FDP-AK7213-V2]');
xlabel('Number of Steps');
ylabel('Time (s)');
grid on

figure
plot(bigGrid, nIters, '-o','LineWidth', 2, 'MarkerSize', 10)
title('Number of Iterations vs Number of Steps [FDP-AK7213-V2]');
xlabel('Number of Steps');
ylabel('Number of Iterations');
grid on

figure
loglog(bigGrid, tSolv./nIters, 'x', 'LineWidth', 2, 'MarkerSize', 15)
hold on
loglog(fineGrid, bestFitLine, 'LineWidth', 2)
xlim([fineGrid(1) fineGrid(end)])
title('Time Per Iteration vs Number of Steps [FDP-AK7213-V2]');
xlabel('Number of Steps');
ylabel('Time Per Iteration (s)');
grid on

fprintf('Line of Best Fit: (%3.6f)x^(%3.6f) + %3.6f\n', eqn.a, eqn.b, eqn.c);
save('SIM_FDP_AK7213_SIM2')
