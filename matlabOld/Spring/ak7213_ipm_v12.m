%% ak7213_ipm_v12
% Matlab based Interior Point Method to solve a Linear Program
% v[1][2] indicates the following
% [1] - Same Arrival Time, Same Deadlines Problem
% [2] - Eliminating t and lambda

% Author: Anand Kasture (ak7213@ic.ac.uk)
% Date: 12 Feb 2017

%% Notation
% m:     scalar number of processors
% l:     scalar number of speed levels
% s:     column vector of speed levels [lx1]
% AP:    column vector of active power datas corresponding to each speed level [lx1]
% IP:    scalar constant for idle power data
% n:     scalar number of tasks
% N:     scalar number of discretization steps
% tau:   column vector of time values (k=0,1,...,N) [(N+1)x1]
% xBar:  column vector containing minimum execution time of tasks [nx1]

rng(2)
close all
clc
clear

% Problem Size
n = 100; N = 10; l = 10; m = 10;

s = (1/l:1/l:1)';
step = 5;
tau = (0:step:step*N)';
xBar = step*rand(n,1)+1;

fprintf('Starting IPM v12 for n=%d; N=%d; l=%d; m=%d\n',n,N,l,m);

% Simplified Dynamic Power Consumption Model: AP(s) = alpha * s^beta
% alpha > 0 ; beta >= 1
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);

%% IPM Solver Setup
nl = n * l;
nd = n + 1 + nl;

x = [xBar, zeros(n,N)];			% find states x for k = 1,...,N
a = zeros(nl,N);            % find inputs a for k = 0,...,N-1
lambda = ones(nd,N);
p = ones(n,N);
t = ones(nd,N);
beta = ones(n,1);

xOpt = [xBar, zeros(n,N)];
aOpt = zeros(nl,N);
lambdaOpt = zeros(nd,N);
pOpt = zeros(n,N);
tOpt = zeros(nd,N);
betaOpt = zeros(n,1);

ipmTol = 1e-4;
minresTol = 1e-6;
ipmMaxIter = 40;
minresMaxIter = 20000;
ipmIter= 1;

% Debugging/Performance Flags
tFullLHS = zeros(ipmMaxIter,1);

tpRHS = zeros(ipmMaxIter,1);
tpBackslash = zeros(ipmMaxIter,1);
tpMinres = zeros(ipmMaxIter,1);

tcRHS = zeros(ipmMaxIter,1);
tcBackslash = zeros(ipmMaxIter,1);
tcMinres = zeros(ipmMaxIter,1);

affAlphaVec = zeros(ipmMaxIter,1);
alphaVec = zeros(ipmMaxIter,1);
dualGap = zeros(ipmMaxIter+1,1);

method = getenv('method');

% Set up LHS components that do not change at each iteration
ts = tic;
d = [ones(n,1); m; zeros(nl,1)];

D1 = kron(speye(n), ones(1,l));
D2 = ones(1,nl);
D3 = -1 * speye(nl);
D = [D1; D2; D3];

B = -kron(step * speye(n), s');

%% Block 0 (k = 0)
rowOff = zeros(3,3);
colOff = zeros(3,3);
dim = [nl n n];
for i = 1:3
    for j = 1:3
        rowOff(i,j) = sum(dim(1:(i-1)));
        colOff(i,j) = sum(dim(1:(j-1)));
    end
end

% row 1
[i,j,s12] = find(B');
i12 = i + rowOff(1,2);
j12 = j + colOff(1,2);

% row 2
[i,j,s21] = find(B);
i21 = i + rowOff(2,1);
j21 = j + colOff(2,1);

% Minus Is
[i,j,s] = find(-1*speye(n));
i32 = i + rowOff(3,2);
j32 = j + colOff(3,2);

i23 = i + rowOff(2,3);
j23 = j + colOff(2,3);

% set up R indices for block 0
r0Rows = kron(ones(nl,1),(1:1:nl)');
r0Cols = kron((1:1:nl)',ones(nl,1));

blk0Rows = [i12; i21; i32; i23];
blk0Cols = [j12; j21; j32; j23];
blk0Vals = [s12; s21; s; s];
% spy(sparse(blk0Rows, blk0Cols, blk0Vals));

outerBlk0Dim = sum(dim);
innerBlk0Dim = outerBlk0Dim - n;

%% Block k (k = 1,2,...,N-1)
rowOff = zeros(4,4);
colOff = zeros(4,4);
dim = [n nl n n];

for i = 1:4
    for j = 1:4
        rowOff(i,j) = sum(dim(1:(i-1)));
        colOff(i,j) = sum(dim(1:(j-1)));
    end
end

outerBlk1Dim = sum(dim);
innerBlk1Dim = outerBlk1Dim - n;

[i,j,s] = find(speye(n));
i13 = i + rowOff(1,3);
j13 = j + colOff(1,3);

i31 = i + rowOff(3,1);
j31 = j + colOff(3,1);

blk1Rows = [blk0Rows + n; i13; i31];
blk1Cols = [blk0Cols + n; j13; j31];
blk1Vals = [blk0Vals; s; s];

nnzVals = length(blk1Vals);

offset1 = innerBlk0Dim * ones((N-1) * nnzVals ,1);
mult = repmat((0:1:N-2), nnzVals,1);
mult = mult(:);
% can also use: mult = kron((0:1:N-2)',ones(nnzVals,1))
offset2 = mult .* (innerBlk1Dim * ones((N-1) * nnzVals ,1));
offsets = offset1 + offset2;

blkKRows = repmat(blk1Rows, N-1, 1) + offsets;
blkKCols = repmat(blk1Cols, N-1, 1) + offsets;
blkKVals = repmat(blk1Vals, N-1, 1);

%% Block N (k = N)
jump = max(offsets);
[i,j,s] = find(speye(n));
i1 = i + rowOff(4,4) + jump;
j1 = j + colOff(4,4) + n + jump;

i2 = i + rowOff(4,4) + n + jump;
j2 = j + colOff(4,4) + jump;

%% Combine!
Rows = [blk0Rows; blkKRows; i1; i2];
Cols = [blk0Cols; blkKCols; j1; j2];
Vals = [blk0Vals; blkKVals; s; s];
%spy(sparse(Rows, Cols, Vals));

%% Set up Row/Col Indices for all Rs
rKRows = r0Rows + rowOff(2,2);
rKCols = r0Cols + colOff(2,2);

% mult = repmat((0:1:N-2), nl^2,1);
% mult = mult(:);
% can also use: mult = kron((0:1:N-2)',ones(nl^2,1))
% offset1 = innerBlk0Dim * ones((N-1) * nl^2 ,1);
% offset2 = mult .* (innerBlk1Dim * ones((N-1) * nl^2 ,1));
% offsets = offset1 + offset2;

offsets = innerBlk0Dim * ones((N-1) * nl^2 ,1) + ...
          kron((0:1:N-2)',ones(nl^2,1)) .* (innerBlk1Dim * ones((N-1) * nl^2 ,1));

RRows = [r0Rows; repmat(rKRows, N-1,1) + offsets];
RCols = [r0Cols; repmat(rKCols, N-1,1) + offsets];
%RVals change at each iteration of the IPM, so set within the while loop
te = toc(ts);
fprintf('Template LHS complete: %3.5fs\n',te);

%% Start IPM iterations
dualGap(1) = (lambda(:)' * t(:))/(nd*N);
while dualGap(ipmIter) > ipmTol && ipmIter <= ipmMaxIter
    fprintf('[%d] Duality Gap: %3.8f\n',ipmIter, dualGap(ipmIter));

    % Construct Full LHS (JACOBIAN) with latest T and Lambda values
    ts = tic;
    RVals = zeros(nl^2,N);
    for k = 1:N
        [~,~,RVals(:,k)] = find( -D' * diag((-t(:,k).^-1) .* lambda(:,k)) * D);
    end
    LHS = sparse( [Rows; RRows], [Cols; RCols], [Vals; RVals(:)] );
    tFullLHS(ipmIter) = toc(ts);
    fprintf('[%d] Full LHS complete: %3.5fs\n',ipmIter, tFullLHS(ipmIter));

    % Construct RHS Predictor
    ts = tic;
    rx = zeros(n,N+1);
    ra = zeros(nl,N);
    rlambda = zeros(nd,N);
    rp = zeros(n,N);
    rt = zeros(nd,N);
    rbeta = zeros(n,1);

    raHat = zeros(nl,N);
    rlambdaHat = zeros(nd,N);

    % rx - not using rx(1)
    for k = 1:N-1
        rx(:,k+1) = -(p(:,k+1) - p(:,k));
    end
    rx(:,N+1) = -(beta - p(:,N));

    % ra, rlambda, rt
    for k = 1:N
        rp(:,k) = -( x(:,k) + B*a(:,k) - x(:,k+1) );
        ra(:,k) =  -( step*kron(ones(n,1),P) + B'*p(:,k) + D'*lambda(:,k) );
        rlambda(:,k) = -( D*a(:,k) - d + t(:,k));
        rt(:,k) = -( lambda(:,k).*t(:,k) );

        invLambda = spdiags(lambda(:,k).^-1 , 0, nd , nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

        rlambdaHat(:,k) = rlambda(:,k) - invLambda * rt(:,k);
        raHat(:,k) = ra(:,k) - D' * invE * rlambdaHat(:,k);
    end

    % rbeta
    rbeta = -x(:,N+1);

    blk0RHS = [raHat(:,1); rp(:,1)];
    blkKRHS = [rx(:,2:N); raHat(:,2:N); rp(:,2:N)];
    blkNRHS = [rx(:,N+1); rbeta];

    PRHS = [blk0RHS; blkKRHS(:); blkNRHS];
    tpRHS(ipmIter) = toc(ts);
    %fprintf('[%d] Predictor RHS complete: %3.5fs\n',ipmIter, tpRHS(ipmIter));


    %% SOLVE PREDICTOR
    if(strcmpi(method,'minres'))
        ts = tic;
        PSOL = minres(LHS, PRHS, minresTol, minresMaxIter);
        te = toc(ts);
        tpMinres(ipmIter) = toc(ts);
        %fprintf('[%d] Predictor Minres: %3.5fs\n',ipmIter, tpMinres(ipmIter));
    else
        ts = tic;
        PSOL = LHS \ PRHS;
        tpBackslash(ipmIter) = toc(ts);
        %fprintf('[%d] Predictor Backslash: %3.5fs\n',ipmIter, tpBackslash(ipmIter));
    end

    affDx = zeros(n,N+1);
    affDa = zeros(nl,N);
    affDlambda = zeros(nd,N);
    affDp = zeros(n,N);
    affDt = zeros(nd,N);
    affDbeta = zeros(n,1);

    % update predictor dx da and dp, and recover dlambda, dt from PSOL
	  % dim = [n nl n n];

    blk0Sol = PSOL(1:innerBlk0Dim);
    blkKSol = reshape(PSOL(innerBlk0Dim+1:innerBlk0Dim + (N-1)*innerBlk1Dim),[innerBlk1Dim,(N-1)]);
    blkNSol = PSOL(end-(2*n)+1:end);

    for k = 1:N
        T = spdiags(t(:,k),0,nd,nd);
        invLambda = spdiags(lambda(:,k).^-1,0,nd,nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

        if(k == 1)
            affDa(:,k) = blk0Sol(1:dim(2));
            affDp(:,k) = blk0Sol(dim(2)+1:sum(dim(2:3)));
        else
            affDx(:,k) = blkKSol(1:dim(1),k-1);
            affDa(:,k) = blkKSol(dim(1)+1:sum(dim(1:2)),k-1);
            affDp(:,k) = blkKSol(sum(dim(1:2))+1:sum(dim(1:3)),k-1);
        end

        affDlambda(:,k) = (invE * rlambdaHat(:,k)) - (invE * D * affDa(:,k));
        affDt(:,k) = (invLambda * rt(:,k)) - (invLambda * T * affDlambda(:,k));
    end

    affDx(:,N+1) = blkNSol(1:dim(1));
    affDbeta = blkNSol(dim(1)+1:end);

    % CORRECTOR RHS
    % find best affine step - % only need to consider the cases where dlambda_aff and dt_aff is negative
    affAlpha = realmax;
    for i = 1:N
        for j = 1:nd
            if( affDlambda(j,i) < 0 )
                affAlpha = min(affAlpha, -lambda(j,i)/affDlambda(j,i));
            end
            if( affDt(j,i) < 0 )
                affAlpha = min(affAlpha, -t(j,i)/affDt(j,i));
            end
        end
    end
    affAlphaVec(ipmIter) = affAlpha;
    fprintf('[%d] Affine Alpha (Step Length): %3.8f\n',ipmIter, affAlpha);

    % calculate sigma = (affDualGap/dualGap)^3
    affDualGap = ( (lambda(:) + (affAlpha * affDlambda(:)))' * (t(:) + (affAlpha * affDt(:))) )/(nd*N);
    fprintf('[%d] Affine Duality Gap: %3.8f\n',ipmIter, affDualGap);

    sigma = (affDualGap/dualGap(ipmIter))^3;
    fprintf('[%d] Sigma: (%3.8f/%3.8f)^3 = %3.8f\n',ipmIter, affDualGap, dualGap(ipmIter),sigma);
    % sigma ~ 0 represents good affine search direction

    ts = tic;
    rx = zeros(n,N+1);
    ra = zeros(nl,N);
    rlambda = zeros(nd,N);
    rp = zeros(n,N);
    rt = -(affDlambda.*affDt) + (sigma * dualGap(ipmIter) * ones(nd, N));
    rbeta = zeros(n,1);

    raHat = zeros(nl,N);
    rlambdaHat = zeros(nd,N);

    for k = 1:N
        invLambda = spdiags(lambda(:,k).^-1 , 0, nd , nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

		rlambdaHat(:,k) = rlambda(:,k) - (invLambda * rt(:,k));
        raHat(:,k) = ra(:,k) - (D' * invE * rlambdaHat(:,k));
    end


    blk0RHS = [raHat(:,1); rp(:,1)];
    blkKRHS = [rx(:,2:N); raHat(:,2:N); rp(:,2:N)];
    blkNRHS = [rx(:,N+1); rbeta];

    CRHS = [blk0RHS; blkKRHS(:); blkNRHS];
    tcRHS(ipmIter) = toc(ts);
    %fprintf('[%d] Corrector RHS complete: %3.5fs\n',ipmIter, tcRHS(ipmIter));

    %% SOLVE CORRECTOR
    if(strcmpi(method,'minres'))
        tic
        CSOL = minres(LHS, CRHS, minresTol, minresMaxIter);
        tcMinres(ipmIter) = toc(ts);
        %fprintf('[%d] Corrector Minres: %3.5fs\n',ipmIter, tcMinres(ipmIter));
    else
        tic
        CSOL = LHS \ CRHS;
        tcBackslash(ipmIter) = toc(ts);
        %fprintf('[%d] Corrector Backslash: %3.5fs\n',ipmIter, tcBackslash(ipmIter));
    end

    ccDx = zeros(n,N+1);
    ccDa = zeros(nl,N);
    ccDlambda = zeros(nd,N);
    ccDp = zeros(n,N);
    ccDt = zeros(nd,N);
    ccDbeta = zeros(n,1);

    % update corrector dx da and dp, and recover dlambda, dt from CSOL
	  % dim = [n nl n n];
    blk0Sol = CSOL(1:innerBlk0Dim);
    blkKSol = reshape(CSOL(innerBlk0Dim+1:innerBlk0Dim + (N-1)*innerBlk1Dim),[innerBlk1Dim,(N-1)]);
    blkNSol = CSOL(end-(2*n)+1:end);

    for k = 1:N
        T = spdiags(t(:,k),0,nd,nd);
        invLambda = spdiags(lambda(:,k).^-1,0,nd,nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

        if(k == 1)
            ccDa(:,k) = blk0Sol(1:dim(2));
            ccDp(:,k) = blk0Sol(dim(2)+1:sum(dim(2:3)));
        else
            ccDx(:,k) = blkKSol(1:dim(1),k-1);
            ccDa(:,k) = blkKSol(dim(1)+1:sum(dim(1:2)),k-1);
            ccDp(:,k) = blkKSol(sum(dim(1:2))+1:sum(dim(1:3)),k-1);
        end

        ccDlambda(:,k) = (invE * rlambdaHat(:,k)) - (invE * D * ccDa(:,k));
        ccDt(:,k) = (invLambda * rt(:,k)) - (invLambda * T * ccDlambda(:,k));

    end

    ccDx(:,N+1) = blkNSol(1:dim(1));
    ccDbeta = blkNSol(dim(1)+1:end);

    %% Determine Search Direction
    Dx = affDx + ccDx;
    Da = affDa + ccDa;
    Dlambda = affDlambda + ccDlambda;
    Dp = affDp + ccDp;
    Dt = affDt + ccDt;
    Dbeta = affDbeta + ccDbeta;

    % find best overall (affine + centering) step size
    alpha = realmax;
    for i = 1:N
        for j = 1:nd
            if( Dlambda(j,i) < 0 )
                alpha = min(alpha, -lambda(j,i)/Dlambda(j,i));
            end
            if( Dt(j,i) < 0 )
                alpha = min(alpha, -t(j,i)/Dt(j,i));
            end
        end
    end
    alphaVec(ipmIter) = alpha;
    fprintf('[%d] Alpha (Step Length): %3.8f\n',ipmIter, alpha);


    %% Update for Next Iteration
    gamma = 1 - (1/(ipmIter+5)^2);
    %gamma = 0.99;

    % Update Using Predictor Corrector Method
    x = x + (gamma * alpha * Dx);
    a = a + (gamma * alpha * Da);
    lambda = lambda + (gamma * alpha * Dlambda);
    p = p + (gamma * alpha * Dp);
    t = t + (gamma * alpha * Dt);
    beta = beta + (gamma * alpha * Dbeta);

    dualGap(ipmIter+1) = ( lambda(:)' * t(:) )/(nd*N);

    if dualGap(ipmIter+1) < min(dualGap(1:ipmIter))
        xOpt = x;
        aOpt = a;
        lambdaOpt = lambda;
        pOpt = p;
        tOpt = t;
        betaOpt = beta;
    end

    fprintf('End of iteration [%d] - DualGap: %3.8f -> %3.8f\n',ipmIter, dualGap(ipmIter), dualGap(ipmIter+1));

    ipmIter = ipmIter+1;
end

%% Analysis
figure
plot(0:1:N,xOpt', 'LineWidth', 2);
title('Same-Arrival-Same-Deadline [n=6; N=3; l=5; m=10]');
xlabel('Time Step [k]');
ylabel('Remaining Time (s)');
grid on
grid minor

figure
spy(LHS)
title('Sparsity of the LHS Matrix: NNZ = 16.94%');
grid on
grid minor
