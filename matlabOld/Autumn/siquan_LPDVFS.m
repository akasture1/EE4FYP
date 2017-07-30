function [cost, a, x, niter] = LPDVFS(m, l, s, AP, IP, N, nj, tau, n, na, xd, tol)
%%
% m: number of processors
% l: number of speed levels
% s: vector of speed levels
% AP: vector of active power datas
% IP: idle power data
% N: number of discretization steps
% nj: a vector containing number of deadlines at each step
% tau: time grid
% n: number of tasks
% na: arrival time of tasks
% xd: minimum execution time of tasks
% tol: tolerance of dual gap
%%
% Note:
% If we have 4 tasks, we set n = 4;
% The minimum execution time is set as xd = [4; 4; 2; 2];
% The arrival time is set as na = [1; 1; 1; 2];
% That means the first three tasks arrive at k = 1
% and the last one task arrives at k = 2
% The deadlines are set as nj = [2; 2;];
% That means the first two tasks have the deadlines at k = 1;
% and the last two tasks have the deadlines at k = 2;
%%
s = s';
Ps = AP'-IP;
tau = tau';
nj = nj';
%5
global TT SSS LLL;
LLL = l;
SSS = s;
niter = 0;
nremain = zeros(N,1);
nremain(1) = n;
for k = 1:N-1
    nremain(k+1) = nremain(k)-nj(k);
end
nm = max(nj);
TT = tau(2:N+1) - tau(1:N);
ss = -s';
nn = n*l+n+1;
rho = 1e80;
%%
x = zeros(n,N);
a = zeros(n*l,N);
p = ones(n,N);
lambda = ones(nn,N);
t = ones(nn,N);
b = ones(nm,N);

% dual gap
smu = 0;
nmu = 0;
for k = 1:N
    smu = smu+lambda(1:nremain(k)*l+nremain(k)+1,k)'*t(1:nremain(k)*l+nremain(k)+1,k);
    nmu = nmu+nremain(k)*l+nremain(k)+1;
end
mu = smu/nmu;

%%
%-------------------------------------------START OF WHILE LOOP--------------------------------------------%
while mu > tol
niter = niter +1;
lt = lambda./t;
tl = t./lambda;

%% Compute RHS for the predictor step
ra = zeros(n*l,N);
rlambda = zeros(nn,N);
rp = zeros(n,N);
rx = zeros(n,N);
rb = zeros(nm,N);
rt = -t.*lambda;
trlambda = zeros(nn,N);
for k = 1:N
    PPs = TT(k)*kron(ones(1,nremain(k)),Ps);
    d = [ones(nremain(k),1);m;zeros(nremain(k)*l,1)];
    rra = ones(nremain(k),1);
    for i = 1:nremain(k)
        rra(i,1) = sum(a((i-1)*l+1:i*l,k));
    end
    rlambda(1:nremain(k)*l+nremain(k)+1,k) = -([rra;sum(a(1:l*nremain(k),k));-a(1:l*nremain(k),k)]-d+t(1:nremain(k)*l+nremain(k)+1,k));
    ra(1:nremain(k)*l,k) = -( PPs' + TT(k)*kron(p(1:nremain(k),k),ss) + kron(lambda(1:nremain(k),k),ones(l,1)) + lambda(nremain(k)+1,k)*ones(nremain(k)*l,1) - lambda(nremain(k)+2:nremain(k)*l+nremain(k)+1,k));
    trlambda(1:nremain(k)*l+nremain(k)+1,k) = rlambda(1:nremain(k)*l+nremain(k)+1,k) - rt(1:nremain(k)*l+nremain(k)+1,k)./lambda(1:nremain(k)*l+nremain(k)+1,k);
    rb(1:nj(k),k)=-x(1:nj(k),k);
end

for i = 1:n
    for k = 1:na(i)-1
        ra((i-n+nremain(k)-1)*l+1:(i-n+nremain(k))*l,k) = ra((i-n+nremain(k)-1)*l+1:(i-n+nremain(k))*l,k) - (rho*TT(k)/tol);
    end
end

for k = 1:N-1
    rx(1:nremain(k),k) = p(1:nremain(k),k)-vertcat(b(1:nj(k),k),p(1:nremain(k+1),k+1));
end
rx(1:nremain(N),N) = p(1:nremain(N),N) - b(1:nj(N),N);

rp(1:nremain(1),1) = -(xd + Btimes(a(1:nremain(1)*l,1),nremain(1),1)-x(1:nremain(1),1));
for k = 2:N
    rp(1:nremain(k),k) = -(x(nj(k-1)+1:nremain(k-1),k-1)+Btimes(a(1:nremain(k)*l,k),nremain(k),k)-x(1:nremain(k),k));
end

%%
deltap = zeros(n,N);
deltaa = zeros(n*l,N);
deltax = zeros(n,N);
deltalambda = zeros(nn,N);
deltat = zeros(nn,N);
deltab = zeros(n,1);
pb = zeros(n,n);
stimesP11 = zeros(n,N);
BtimesP3 = zeros(n,N);
sumP11 = zeros(n,N);
r = zeros(n,1);
hatrp = zeros(n,N);
hatra = zeros(n*l,N);
hatrx = zeros(n,N);
hhatrx = zeros(n*l,N);

%%
hatrp(1:nremain(N),N) = rp(1:nremain(N),N) + rb(1:nj(N),N);
hatrx(1:nremain(N),N) = rx(1:nremain(N),N);
for k = 1:N-1
    kk = N - k;
    hatrp(1:nremain(kk),kk) = rp(1:nremain(kk),kk);
    hatrp(1:nj(kk),kk) = hatrp(1:nj(kk),kk) + rb(1:nj(kk),kk);
    hatrp(nj(kk)+1:nremain(kk),kk) = hatrp(nj(kk)+1:nremain(kk),kk) + hatrp(1:nremain(kk+1),kk+1);
    hatrx(1:nremain(kk),kk) = rx(1:nremain(kk),kk) + vertcat(zeros(nj(kk),1),hatrx(1:nremain(kk+1),kk+1));
end

r = r - hatrp(1:n,1);

for k = 1:N
ll = lt(1:nremain(k)*l+nremain(k)+1,k).*trlambda(1:nremain(k)*l+nremain(k)+1,k);
hatra(1:nremain(k)*l,k) = ra(1:l*nremain(k),k) + kron(ll(1:nremain(k),1),ones(l,1))+ll(nremain(k)+1,1)*ones(nremain(k)*l,1)-ll(nremain(k)+2:nremain(k)*l+nremain(k)+1,1);
hhatrx(1:nremain(k)*l,k) = TT(k)*kron(hatrx(1:nremain(k),k),ss);
S1 = lt(1:nremain(k),k);
S2 = lt(nremain(k)+1,k);
S3 = lt(nremain(k)+2:nn,k);
P1 = zeros(nremain(k)*l,1);
P2 = zeros(nremain(k)*l,1);
P4 = zeros(nremain(k)*l,1);
for i = 1:nremain(k)
    s1 = S1(i);
    s3 = S3((i-1)*l+1:i*l);
    hatr = hatra((i-1)*l+1:i*l,k);
    hhatr = hhatrx((i-1)*l+1:i*l,k);
    is3 = 1./s3;
    p1 = sum(is3);
    p2 = s1/(1+s1*p1);
    p3 = is3.*hatr;
    p4 = is3.*hhatr;
    P1((i-1)*l+1:i*l) = p3 - is3*p2*sum(p3);
    P2((i-1)*l+1:i*l) = is3*(1-p2*p1);
    P4((i-1)*l+1:i*l) = p4 - is3*p2*sum(p4);
    pp3 = TT(k)*is3.*ss;
    P11 = pp3 - is3*p2*sum(pp3);
    stimesP11(i,k) = -TT(k)*s*P11;
    sumP11(i,k) = sum(P11);
end
P3 = S2/(1+S2*sum(P2))*P2;
r = r + vertcat(zeros(n-nremain(k),1),Btimes(P1 - P3*sum(P1),nremain(k),k));
r = r + vertcat(zeros(n-nremain(k),1),Btimes(P4 - P3*sum(P4),nremain(k),k));
BtimesP3(1:nremain(k),k) = Btimes(P3,nremain(k),k);
pb = pb - blk(zeros(n-nremain(k),n-nremain(k)),diag(stimesP11(1:nremain(k),k))-BtimesP3(1:nremain(k),k)*sumP11(1:nremain(k),k)');
end

%% solve Deltab = -sum(BR^-1B') \ r with LU factorization
[L,U] = lu(pb);
deltab(1,1) = -r(1,1);
for i = 2:n
    tdeltab = 0;
    for j = 1:i-1
        tdeltab = tdeltab + L(i,j) * deltab(j,1);
    end
    deltab(i,1) = -r(i,1) - tdeltab;
end

deltab(n,1) = deltab(n,1)/U(n,n);
for ii = 1:n-1
    i = n - ii;
    tdeltab = 0;
    for j = i+1: n
        tdeltab = tdeltab + U(i,j) * deltab(j,1);
    end
    deltab(i,1) = (deltab(i,1) - tdeltab)/U(i,i);
end

%% Compute search direction of the predictor step
deltap(1:nremain(N),N) = deltab(n-nremain(N)+1:n,1) - rx(1:nremain(N),N);
for k = 1:N-1
    kk = N - k;
    deltap(1:nremain(kk),kk) = vertcat(deltab(n-nremain(kk)+1:n-nremain(kk)+nj(kk),1),deltap(1:nremain(kk+1),kk+1)) - rx(1:nremain(kk),kk);
end

for k = 1:N
ll = lt(1:nremain(k)*l+nremain(k)+1,k).*trlambda(1:nremain(k)*l+nremain(k)+1,k);
Bp = TT(k)*kron(deltap(:,k),s');
hatra(1:nremain(k)*l,k) = ra(1:l*nremain(k),k) + kron(ll(1:nremain(k),1),ones(l,1))+ll(nremain(k)+1,1)*ones(nremain(k)*l,1)-ll(nremain(k)+2:nremain(k)*l+nremain(k)+1,1)+Bp(1:l*nremain(k),1);
S1 = lt(1:nremain(k),k);
S2 = lt(nremain(k)+1,k);
S3 = lt(nremain(k)+2:nn,k);
P1 = zeros(nremain(k)*l,1);
P2 = zeros(nremain(k)*l,1);
for i = 1:nremain(k)
    s1 = S1(i);
    s3 = S3((i-1)*l+1:i*l);
    hatr = hatra((i-1)*l+1:i*l,k);
    is3 = 1./s3;
    p1 = sum(is3);
    p2 = s1/(1+s1*p1);
    p3 = is3.*hatr;
    P1((i-1)*l+1:i*l) = p3 - is3*p2*sum(p3);
    P2((i-1)*l+1:i*l) = is3*(1-p2*p1);
end
P3 = S2/(1+S2*sum(P2))*P2;
deltaa(1:nremain(k)*l,k) = P1 - P3*sum(P1);

dla = ones(nremain(k),1);
for i = 1:nremain(k)
    dla(i,1) = sum(deltaa((i-1)*l+1:i*l,k));
end
deltalambda(1:nremain(k)*l+nremain(k)+1,k) = lt(1:nremain(k)*l+nremain(k)+1,k).*vertcat(dla,sum(deltaa(1:nremain(k)*l,k)),-deltaa(1:l*nremain(k),k)) -trlambda(1:nremain(k)*l+nremain(k)+1,k).*lt(1:nremain(k)*l+nremain(k)+1,k);

deltat(1:nremain(k)*l+nremain(k)+1,k) = rt(1:nremain(k)*l+nremain(k)+1,k)./lambda(1:nremain(k)*l+nremain(k)+1,k) - tl(1:nremain(k)*l+nremain(k)+1,k).*deltalambda(1:nremain(k)*l+nremain(k)+1,k);

end

deltax(1:nremain(1),1) = Btimes(deltaa(1:l*nremain(1),1),nremain(1),1) - rp(1:nremain(1),1);
for k = 2:N
deltax(1:nremain(k),k) = Btimes(deltaa(1:l*nremain(k),k),nremain(k),k) - rp(1:nremain(k),k) + deltax(nj(k-1)+1:nremain(k-1),k-1);
end
%%
k = 1;
aaaff = 10000;
for i = 1:nremain(k)*l+nremain(k)+1
    if lambda(i,k)/deltalambda(i,k)<0
        aaaff = min(aaaff, -lambda(i,k)./deltalambda(i,k));
    end
    if t(i,k)/deltat(i,k)<0
        aaaff = min(aaaff, -t(i,k)/deltat(i,k));
    end
end
for k = 2:N
for i = 1:nremain(k)*l+nremain(k)+1
    if lambda(i,k)/deltalambda(i,k)<0
        aaaff = min(aaaff, -lambda(i,k)./deltalambda(i,k));
    end
    if t(i,k)/deltat(i,k)<0
        aaaff = min(aaaff, -t(i,k)/deltat(i,k));
    end
end
end

smu = 0;
nmu = 0;
for k = 1:N
    smu = smu+(lambda(1:nremain(k)*l+nremain(k)+1,k)+aaaff.*deltalambda(1:nremain(k)*l+nremain(k)+1,k))'*(t(1:nremain(k)*l+nremain(k)+1,k)+aaaff.*deltat(1:nremain(k)*l+nremain(k)+1,k));
    nmu = nmu+nremain(k)*l+nremain(k)+1;
end
muaaf = smu/nmu;
sigma = (muaaf/mu)^3;
%%
Da = deltaa;
Dlambda = deltalambda;
Dp = deltap;
Dx = deltax;
Dt = deltat;
Db = deltab;

%% Compute RHS for the centering step
rx = zeros(n,N);
ra = zeros(n*l,N);
rp = zeros(n,N);
rt = -(deltalambda.*deltat)+sigma*mu.*ones(nn,N);
rlambda = zeros(nn,N);
trlambda = rlambda - rt./lambda;
rb = zeros(nm,N);

%%
deltap = zeros(n,N);
deltaa = zeros(n*l,N);
deltax = zeros(n,N);
deltalambda = zeros(nn,N);
deltat = zeros(nn,N);
deltab = zeros(n,1);
stimesP11 = zeros(n,N);
BtimesP3 = zeros(n,N);
sumP11 = zeros(n,N);
r = zeros(n,1);
hatrp = zeros(n,N);
hatra = zeros(n*l,N);
hatrx = zeros(n,N);
hhatrx = zeros(n*l,N);

%%
hatrp(1:nremain(N),N) = rp(1:nremain(N),N) + rb(1:nj(N),N);
hatrx(1:nremain(N),N) = rx(1:nremain(N),N);
for k = 1:N-1
    kk = N - k;
    hatrp(1:nremain(kk),kk) = rp(1:nremain(kk),kk);
    hatrp(1:nj(kk),kk) = hatrp(1:nj(kk),kk) + rb(1:nj(kk),kk);
    hatrp(nj(kk)+1:nremain(kk),kk) = hatrp(nj(kk)+1:nremain(kk),kk) + hatrp(1:nremain(kk+1),kk+1);
    hatrx(1:nremain(kk),kk) = rx(1:nremain(kk),kk) + vertcat(zeros(nj(kk),1),hatrx(1:nremain(kk+1),kk+1));
end

r = r - hatrp(1:n,1);

%%
for k = 1:N
ll = lt(1:nremain(k)*l+nremain(k)+1,k).*trlambda(1:nremain(k)*l+nremain(k)+1,k);
hatra(1:nremain(k)*l,k) = ra(1:l*nremain(k),k) + kron(ll(1:nremain(k),1),ones(l,1))+ll(nremain(k)+1,1)*ones(nremain(k)*l,1)-ll(nremain(k)+2:nremain(k)*l+nremain(k)+1,1);
hhatrx(1:nremain(k)*l,k) = TT(k)*kron(hatrx(1:nremain(k),k),ss);
S1 = lt(1:nremain(k),k);
S2 = lt(nremain(k)+1,k);
S3 = lt(nremain(k)+2:nn,k);
P1 = zeros(nremain(k)*l,1);
P2 = zeros(nremain(k)*l,1);
P4 = zeros(nremain(k)*l,1);
for i = 1:nremain(k)
    s1 = S1(i);
    s3 = S3((i-1)*l+1:i*l);
    hatr = hatra((i-1)*l+1:i*l,k);
    hhatr = hhatrx((i-1)*l+1:i*l,k);
    is3 = 1./s3;
    p1 = sum(is3);
    p2 = s1/(1+s1*p1);
    p3 = is3.*hatr;
    p4 = is3.*hhatr;
    P1((i-1)*l+1:i*l) = p3 - is3*p2*sum(p3);
    P2((i-1)*l+1:i*l) = is3*(1-p2*p1);
    P4((i-1)*l+1:i*l) = p4 - is3*p2*sum(p4);
    pp3 = TT(k)*is3.*ss;
    P11 = pp3 - is3*p2*sum(pp3);
    stimesP11(i,k) = -TT(k)*s*P11;
    sumP11(i,k) = sum(P11);
end
P3 = S2/(1+S2*sum(P2))*P2;
r = r + vertcat(zeros(n-nremain(k),1),Btimes(P1 - P3*sum(P1),nremain(k),k));
r = r + vertcat(zeros(n-nremain(k),1),Btimes(P4 - P3*sum(P4),nremain(k),k));
BtimesP3(1:nremain(k),k) = Btimes(P3,nremain(k),k);

end

%% Compute search direction of the predictor step
deltab(1,1) = -r(1,1);
for i = 2:n
    tdeltab = 0;
    for j = 1:i-1
        tdeltab = tdeltab + L(i,j) * deltab(j,1);
    end
    deltab(i,1) = -r(i,1) - tdeltab;
end

deltab(n,1) = deltab(n,1)/U(n,n);
for ii = 1:n-1
    i = n - ii;
    tdeltab = 0;
    for j = i+1: n
        tdeltab = tdeltab + U(i,j) * deltab(j,1);
    end
    deltab(i,1) = (deltab(i,1) - tdeltab)/U(i,i);
end
deltap(1:nremain(N),N) = deltab(n-nremain(k)+1:n,1) - rx(1:nremain(N),N);
for k = 1:N-1
    kk = N - k;
    deltap(1:nremain(kk),kk) = vertcat(deltab(n-nremain(kk)+1:n-nremain(kk)+nj(kk),1),deltap(1:nremain(kk+1),kk+1)) - rx(1:nremain(kk),kk);
end

for k = 1:N
ll = lt(1:nremain(k)*l+nremain(k)+1,k).*trlambda(1:nremain(k)*l+nremain(k)+1,k);
Bp = TT(k)*kron(deltap(:,k),s');
hatra(1:nremain(k)*l,k) = ra(1:l*nremain(k),k) + kron(ll(1:nremain(k),1),ones(l,1))+ll(nremain(k)+1,1)*ones(nremain(k)*l,1)-ll(nremain(k)+2:nremain(k)*l+nremain(k)+1,1)+Bp(1:l*nremain(k),1);
S1 = lt(1:nremain(k),k);
S2 = lt(nremain(k)+1,k);
S3 = lt(nremain(k)+2:nn,k);
P1 = zeros(nremain(k)*l,1);
P2 = zeros(nremain(k)*l,1);
for i = 1:nremain(k)
    s1 = S1(i);
    s3 = S3((i-1)*l+1:i*l);
    hatr = hatra((i-1)*l+1:i*l,k);
    is3 = 1./s3;
    p1 = sum(is3);
    p2 = s1/(1+s1*p1);
    p3 = is3.*hatr;
    P1((i-1)*l+1:i*l) = p3 - is3*p2*sum(p3);
    P2((i-1)*l+1:i*l) = is3*(1-p2*p1);
end
P3 = S2/(1+S2*sum(P2))*P2;
deltaa(1:nremain(k)*l,k) = P1 - P3*sum(P1);

dla = ones(nremain(k),1);
for i = 1:nremain(k)
    dla(i,1) = sum(deltaa((i-1)*l+1:i*l,k));
end
deltalambda(1:nremain(k)*l+nremain(k)+1,k) = lt(1:nremain(k)*l+nremain(k)+1,k).*vertcat(dla,sum(deltaa(1:nremain(k)*l,k)),-deltaa(1:l*nremain(k),k)) -trlambda(1:nremain(k)*l+nremain(k)+1,k).*lt(1:nremain(k)*l+nremain(k)+1,k);

deltat(1:nremain(k)*l+nremain(k)+1,k) = rt(1:nremain(k)*l+nremain(k)+1,k)./lambda(1:nremain(k)*l+nremain(k)+1,k) - tl(1:nremain(k)*l+nremain(k)+1,k).*deltalambda(1:nremain(k)*l+nremain(k)+1,k);

end

deltax(1:nremain(1),1) = Btimes(deltaa(1:l*nremain(1),1),nremain(1),1) - rp(1:nremain(1),1);
for k = 2:N
deltax(1:nremain(k),k) = Btimes(deltaa(1:l*nremain(k),k),nremain(k),k) - rp(1:nremain(k),k) + deltax(nj(k-1)+1:nremain(k-1),k-1);
end
%%
Dx = deltax + Dx;
Da = deltaa + Da;
Dp = deltap + Dp;
Dlambda = deltalambda + Dlambda;
Dt = deltat + Dt;
Db = deltab + Db;

%% Compute step length
k = 1;
aam = 1;
for i = 1:nremain(k)*l+nremain(k)+1
    if lambda(i,k)/Dlambda(i,k)<0
        aam = min(aam, -lambda(i,k)./Dlambda(i,k));
    end
    if t(i,k)/Dt(i,k)<0
        aam = min(aam, -t(i,k)/Dt(i,k));
    end
end
for k = 2:N
for i = 1:nremain(k)*l+nremain(k)+1
    if lambda(i,k)/Dlambda(i,k)<0
        aam = min(aam, -lambda(i,k)./Dlambda(i,k));
    end
    if t(i,k)/Dt(i,k)<0
        aam = min(aam, -t(i,k)/Dt(i,k));
    end
end
end
aa = 0.99*aam;

%%
x = x + aa.*Dx;
a = a + aa.*Da;
p = p + aa.*Dp;
lambda = lambda + aa.*Dlambda;
t = t + aa.*Dt;
b(1:nj(1),1) = b(1:nj(1),1) + aa.*Db(1:nj(1),1);
for k = 2:N
b(1:nj(k),k) = b(1:nj(k),k) + aa.*Db(n-nremain(k)+1:n-nremain(k)+nj(k),1);
end

if niter == 100 % Max number of iterations
    break;
end

smu = 0;
nmu = 0;
for k = 1:N
    smu = smu+lambda(1:nremain(k)*l+nremain(k)+1,k)'*t(1:nremain(k)*l+nremain(k)+1,k);
    nmu = nmu+nremain(k)*l+nremain(k)+1;
end
mu = smu/nmu;

end
%-------------------------------------------END OF WHILE LOOP--------------------------------------------%


%% Compute cost function
cost = 0;
for k = 1:N
    cost = cost + TT(k)*kron(ones(1,nremain(k)),Ps)*a(1:l*nremain(k),k);
end

function Bx = Btimes(a,n,k)
global TT SSS LLL;
Bx = zeros(n,size(a,2));
for i = 1:n
    Bx(i,:) = -TT(k)*SSS*a(LLL*(i-1)+1:i*LLL,:);
end
