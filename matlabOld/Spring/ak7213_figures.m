close all
clear all

%%
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
%%

m = 100;

%% Figure1: Number of Tasks vs Time per Iteration
l = 10; 
s = (1/l:1/l:1)';

% Simplified Dynamic Power Consumption Model: AP(s) = alpha * s^beta
% alpha > 0 ; beta >= 1
AP = 1000*((1/l:1/l:1).^2)';
IP = 40;
N = 100;
tau = (0:5:5*N)';
T_linprog = zeros(10,1);

ii = 0;
nUpper=2000;
nGrid = 200:200:nUpper;

for n = nGrid
  ii = ii + 1;
  na = ones(n,1); 
  nj = n/N*ones(n,1); 
  xd = 5*rand(n,1);
  [solution, fEval, niter, timeperiter] = ak7213_linp(m, l, s, AP, IP, N, nj, tau, n, na, xd);
  T_linprog(ii) = timeperiter;
end

xx = 1:nUpper;
eqn = fit((nGrid)',T_linprog,'power2');
linprog_fit = eqn.a * xx.^eqn.b + eqn.c;

figure (1)
plot(nGrid,T_linprog,'o-', xx,linprog_fit,'-');
grid on;
grid minor;
xlabel('Number of Tasks n','FontSize',24);
ylabel('Time per Iteration (s)','FontSize',24);
title('Time per Iteration (linprog) vs Number of Tasks','FontSize',24)
set(gca,'fontsize',20)

figure (2)
loglog(nGrid, T_linprog, 'o-');
grid on;
grid minor;
xlabel('Number of Tasks n','FontSize',24);
ylabel('Time per Iteration (s)','FontSize',24);
title('Time per Iteration (linprog) vs Number of Tasks','FontSize',24)
set(gca,'fontsize',20)