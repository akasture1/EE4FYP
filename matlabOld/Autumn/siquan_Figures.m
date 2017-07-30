close all
clear
rng('default')
%%
tol = 1e-5;
m = 100; % number of processors

%% Figure 1 & 4
l = 10; % number of speed levels
s = (1/l:1/l:1)'; % speed levels
AP = 1000*((1/l:1/l:1).^(0.5))'; % active power at each level
IP = 40; % idle power
N = 2; % number of steps in time grid
tau = 0:5:5*N; % time grid
tau = tau';

T_linprog = zeros(10,1);
T_solver = zeros(10,1);
ii = 0;

n=4;
na = ones(n,1); % arrival
nj = n/N*ones(n,1); % deadlines
xd = rand(n,1); % minimum execution time
[solution, fEval, niter, timeperiter] = ak7213_linp(m, l, s, AP, IP, N, nj, tau, n, na, xd);
%[cost, a, x, niter] = LPDVFS(m, l, s, AP, IP, N, nj, tau, n, na, xd, tol);

for n = 200:200:2000 % number of tasks
ii = ii + 1;
na = ones(n,1); % arrival
nj = n/N*ones(n,1); % deadlines
xd = rand(n,1); % minimum execution time
[out, Cost, nitern, timeperiter] = linp(m, l, s, AP, IP, N, nj, tau, n, na, xd);
T_linprog(ii) = timeperiter;
ts = tic;
[cost, a, x, niter] = LPDVFS(m, l, s, AP, IP, N, nj, tau, n, na, xd, tol);
te = toc(ts);
T_solver(ii) = te/niter;
end
fit_linprog = fit((200:200:2000)',T_linprog,'power2');
xx = 1:2000;
figure (1)
plot(200:200:2000,T_linprog,'o',xx,fit_linprog.a*xx.^fit_linprog.b+fit_linprog.c,'-');
grid on;
xlabel('Number of Tasks');
ylabel('Time per Iteration');
fit_solver = fit((200:200:2000)',T_solver,'power2');
figure (4)
plot(200:200:2000,T_solver,'o',xx,fit_solver.a*xx.^fit_solver.b+fit_solver.c,'-');
grid on;
xlabel('Number of Tasks');
ylabel('Time per Iteration');

%% Figure 2 & 5
IP = 40; % idle power
N = 50; % number of steps in time grid
tau = 0:5:5*N; % time grid
tau = tau';
n = 50; % number of tasks
na = ones(n,1); % arrival
nj = n/N*ones(n,1); % deadlines
xd = rand(n,1); % minimum execution time

T_linprog = zeros(10,1);
T_solver = zeros(10,1);
ii = 0;
for l = 100:100:1000 % number of speed levels  
s = (1/l:1/l:1)'; % speed levels
AP = 1000*((1/l:1/l:1).^(0.5))'; % active power at each level
ii = ii + 1;
[out, Cost, niter, timeperiter] = linp(m, l, s, AP, IP, N, nj, tau, n, na, xd);
T_linprog(ii) = timeperiter;
ts = tic;
[cost, a, x, niter] = LPDVFS(m, l, s, AP, IP, N, nj, tau, n, na, xd, tol);
te = toc(ts);
T_solver(ii) = te/niter;
end
fit_linprog = fit((100:100:1000)',T_linprog,'power2');
xx = 1:1000;
figure (2)
plot(100:100:1000,T_linprog,'o',xx,fit_linprog.a*xx.^fit_linprog.b+fit_linprog.c,'-');
grid on;
xlabel('Number of Speed Levels');
ylabel('Time per Iteration');
fit_solver = fit((100:100:1000)',T_solver,'power2');
figure (5)
plot(100:100:1000,T_solver,'o',xx,fit_solver.a*xx.^fit_solver.b+fit_solver.c,'-');
grid on;
xlabel('Number of Speed Levels');
ylabel('Time per Iteration');

%% Figure 3 & 6
IP = 40; % idle power
n = 100; % number of tasks
l = 10; % number of speed levels
s = (1/l:1/l:1)'; % speed levels
AP = 1000*((1/l:1/l:1).^(0.5))'; % active power at each level
na = ones(n,1); % arrival
xd = rand(n,1); % minimum execution time

T_linprog = zeros(6,1);
T_solver = zeros(6,1);
ii = 0;
for N = [5,10,20,25,50,100] % number of steps in time grid 
tau = 0:5:5*N; % time grid
tau = tau';
nj = n/N*ones(n,1); % deadlines
ii = ii + 1;
[out, Cost, niter, timeperiter] = linp(m, l, s, AP, IP, N, nj, tau, n, na, xd);
T_linprog(ii) = timeperiter;
ts = tic;
[cost, a, x, niter] = LPDVFS(m, l, s, AP, IP, N, nj, tau, n, na, xd, tol);
te = toc(ts);
T_solver(ii) = te/niter;
end
fit_linprog = fit([5;10;20;25;50;100],T_linprog,'power2');
xx = 1:100;
figure (3)
plot([5;10;20;25;50;100],T_linprog,'o',xx,fit_linprog.a*xx.^fit_linprog.b+fit_linprog.c,'-');
grid on;
xlabel('Number of Steps');
ylabel('Time per Iteration');
fit_solver = fit([5;10;20;25;50;100],T_solver,'power2');
figure (6)
plot([5;10;20;25;50;100],T_solver,'o',xx,fit_solver.a*xx.^fit_solver.b+fit_solver.c,'-');
grid on;
xlabel('Number of Steps');
ylabel('Time per Iteration');







