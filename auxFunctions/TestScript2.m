%% Function Argument Definitions
% n:          scalar number of tasks
% N:          scalar number of discrete time steps
% l:          scalar number of speed levels
% m:          scalar number of processors
% s:          column vector of speed levels [1xl]
% step:       fixed time step between each point on the time grid
% P:          column vector of net power consumption values corresponding to each speed level [lx1]
% xBar:       column vector containing minimum execution time of tasks [nx1]
% ipmTol:
% ipmMaxIter:
% solver:
% solverTol:
rng(2)
close all
clc
clear

%% Problem Size
n = 4; N = 2; l = 5; m = 10;
s = (1/l:1/l:1)';
step = 5;
P = 10*((1/l:1/l:1).^2)';
P = P + mean(P);
xBar = step*rand(n,1)+1;
ipmTol = 1e-4;
ipmMaxIter = 40;
solver = 1;
solverTol = 1e-6;

fprintf('Starting FDP_AK7213_V2 for n=%d; N=%d; l=%d; m=%d\n',n,N,l,m);
%% IPM Solver Setup
nl = n * l;
nd = n + 1 + nl;
ipmIter = 1;

x = step*zeros(n,N);         % k = 1,...,N
a = zeros(nl,N);             % k = 0,...,N-1
lambda = ones(nd,N);        % k = 0,...,N-1
p = ones(n,N);              % k = 0,...,N-1
t = ones(nd,N);             % k = 0,...,N-1
beta = ones(n,1);           % k = 1,...,N


%% Debugging/Performance Flags
alphaVec = zeros(ipmMaxIter,1);
sigmaVec = zeros(ipmMaxIter,1);
dualGap = zeros(ipmMaxIter+1,1);

%% Constant LHS Components
d = [ones(n,1); m; zeros(nl,1)];

D1 = kron(speye(n), ones(1,l));
D2 = ones(1,nl);
D3 = -1 * speye(nl);
D = [D1; D2; D3];

B = -kron(step * speye(n), s');

%% Construct the LHS Template
% Block 0 (k = 0)
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

%% Start IPM iterations
dualGap(1) = (lambda(:)' * t(:))/(nd*N);
while dualGap(ipmIter) > ipmTol && ipmIter <= ipmMaxIter
    fprintf('[%d] Duality Gap: %3.8f\n',ipmIter, dualGap(ipmIter));

    % Construct Full LHS (JACOBIAN) with latest T and Lambda values
    RVals = zeros(nl^2,N);
    for k = 1:N
        [~,~,RVals(:,k)] = find( -D' * diag((-t(:,k).^-1) .* lambda(:,k)) * D);
    end
    LHS = sparse( [Rows; RRows], [Cols; RCols], [Vals; RVals(:)] );

    %% Construct Predictor RHS
    rx = zeros(n,N);          % k = 1,...,N
    ra = zeros(nl,N);         % k = 0,...,N-1
    rlambda = zeros(nd,N);    % k = 0,...,N-1
    rp = zeros(n,N);          % k = 0,...,N-1
    rt = zeros(nd,N);         % k = 0,...,N-1
    rbeta = zeros(n,1);

    rlambdaHat = zeros(nd,N); % k = 0,...,N-1
    raHat = zeros(nl,N);      % k = 0,...,N-1

    % rx
    for k = 1:N-1
        rx(:,k) = -( -p(:,k) + p(:,k+1) );
    end
    rx(:,N) = -( -p(:,N) + beta );

    % ra
    for k = 1:N
        ra(:,k) =  -( step*kron(ones(n,1),P) + B'*p(:,k) + D'*lambda(:,k) );
    end

    % rlambda
    for k = 1:N
        rlambda(:,k) = -( D*a(:,k) - d + t(:,k) );
    end

    % rp
    rp(:,1) = -( xBar + B*a(:,1) - x(:,1) );
    for k = 1:N-1
        rp(:,k+1) = -( x(:,k) + B*a(:,k+1) - x(:,k+1) );
    end

    % rt
    for k = 1:N
        rt(:,k) = -( lambda(:,k).*t(:,k) );
    end

    % rbeta
    rbeta = -x(:,N);

    % rlambdaHat; raHat
    for k = 1:N
        rlambdaHat(:,k) = rlambda(:,k) - ((lambda(:,k).^-1) .* rt(:,k));
        raHat(:,k) = ra(:,k) + (D' * (t(:,k).^-1 .* lambda(:,k) .* rlambdaHat(:,k)));
    end

    % rbeta
    rbeta = -x(:,N);

    blk0RHS = [raHat(:,1); rp(:,1)];
    blkKRHS = [rx(:,1:N-1); raHat(:,2:N); rp(:,2:N)];
    blkNRHS = [rx(:,N); rbeta];

    PRHS = [blk0RHS; blkKRHS(:); blkNRHS];

    %% Solve Predictor Problem
    switch solver
      case 1
        PSOL = LHS \ PRHS;
      case 2
        PSOL = minres(LHS, PRHS, solverTol, solverMaxIter);
      case 3
        PSOL = cgs(LHS, PRHS, solverTol, solverMaxIter);
    end

    %% Set Affine Parameters from Predictor Solution
    affDx = zeros(n,N);        % k = 1,...,N
    affDa = zeros(nl,N);       % k = 0,...,N-1
    affDlambda = zeros(nd,N);  % k = 0,...,N-1
    affDp = zeros(n,N);        % k = 0,...,N-1
    affDt = zeros(nd,N);       % k = 0,...,N-1
    affDbeta = zeros(n,1);

	  % dim = [n nl n n];
    blk0Sol = PSOL(1:innerBlk0Dim);
    blkKSol = reshape(PSOL(innerBlk0Dim+1:innerBlk0Dim + (N-1)*innerBlk1Dim),[innerBlk1Dim,(N-1)]);
    blkNSol = PSOL(end-(2*n)+1:end);

    affDx(:,1:N-1) = blkKSol(1:dim(1),:);
    affDx(:,N) = blkNSol(1:dim(1));

    affDa(:,1) = blk0Sol(1:dim(2));
    affDa(:,2:N) = blkKSol(dim(1)+1:sum(dim(1:2)),:);

    affDp(:,1) = blk0Sol(dim(2)+1:end);
    affDp(:,2:N) = blkKSol(sum(dim(1:2))+1:sum(dim(1:3)),:);

    affDbeta = blkNSol(dim(1)+1:end);

    for k = 1:N
        affDlambda(:,k) = (t(:,k).^-1 .* lambda(:,k)) .* (-rlambdaHat(:,k) + D*affDa(:,k));
        affDt(:,k) = lambda(:,k).^-1 .* (rt(:,k) - t(:,k) .* affDlambda(:,k));
    end

    %% Calculate Affine Step Length
    % only need to consider the cases where affDlambda and affDt is negative
    affAlpha = 1;
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

    %% Calculate Sigma
    % sigma ~ 0 indicates that the affine dir. is a good search dir.
    affDualGap = ( (lambda(:) + (affAlpha * affDlambda(:)))' * (t(:) + (affAlpha * affDt(:))) )/(nd*N);
    fprintf('[%d] Affine Duality Gap: %3.8f\n',ipmIter, affDualGap);

    sigma = (affDualGap/dualGap(ipmIter))^3;
    sigmaVec(ipmIter) = sigma;
    fprintf('[%d] Sigma: (%3.8f/%3.8f)^3 = %3.8f\n',ipmIter, affDualGap, dualGap(ipmIter),sigma);

    %% Construct Corrector RHS
    rx = zeros(n,N);            % k = 1,...,N
    ra = zeros(nl,N);           % k = 0,...,N-1
    rlambda = zeros(nd,N);      % k = 0,...,N-1
    rp = zeros(n,N);            % k = 0,...,N-1
    rt = -(affDlambda.*affDt) + (sigma * dualGap(ipmIter) * ones(nd, N));       % k = 0,...,N-1
    rbeta = zeros(n,1);         % k = 1,...,N

    rlambdaHat = zeros(nd,N);   % k = 0,...,N-1
    raHat = zeros(nl,N);        % k = 0,...,N-1

    for k = 1:N
        rlambdaHat(:,k) = rlambda(:,k) - (lambda(:,k).^-1) .* rt(:,k);
        raHat(:,k) = ra(:,k) + D' * (t(:,k).^-1 .* lambda(:,k) .* rlambdaHat(:,k));
    end

    blk0RHS = [raHat(:,1); rp(:,1)];
    blkKRHS = [rx(:,1:N-1); raHat(:,2:N); rp(:,2:N)];
    blkNRHS = [rx(:,N); rbeta];

    CRHS = [blk0RHS; blkKRHS(:); blkNRHS];

    %% Solve Corrector Problem
    switch solver
      case 1
        CSOL = LHS \ CRHS;
      case 2
        CSOL = minres(LHS, CRHS, solverTol, solverMaxIter);
      case 3
        CSOL = cgs(LHS, CRHS, solverTol, solverMaxIter);
    end

    %% Set Corrector Parameters from Predictor Solution
    ccDx = zeros(n,N);        % k = 1,...,N
    ccDa = zeros(nl,N);       % k = 0,...,N-1
    ccDlambda = zeros(nd,N);  % k = 0,...,N-1
    ccDp = zeros(n,N);        % k = 0,...,N-1
    ccDt = zeros(nd,N);       % k = 0,...,N-1
    ccDbeta = zeros(n,1);     % k = 1,...,N

    % dim = [n nl nd n nd n];
    blk0Sol = CSOL(1:innerBlk0Dim);
    blkKSol = reshape(CSOL(innerBlk0Dim+1:innerBlk0Dim + (N-1)*innerBlk1Dim),[innerBlk1Dim,(N-1)]);
    blkNSol = CSOL(end-(2*n)+1:end);

    ccDx(:,1:N-1) = blkKSol(1:dim(1),:);
    ccDx(:,N) = blkNSol(1:dim(1));

    ccDa(:,1) = blk0Sol(1:dim(2));
    ccDa(:,2:N) = blkKSol(dim(1)+1:sum(dim(1:2)),:);

    ccDp(:,1) = blk0Sol(dim(2)+1:end);
    ccDp(:,2:N) = blkKSol(sum(dim(1:2))+1:sum(dim(1:3)),:);

    ccDbeta = blkNSol(dim(1)+1:end);

    for k = 1:N
        ccDlambda(:,k) = (t(:,k).^-1 .* lambda(:,k)) .* (-rlambdaHat(:,k) + D*ccDa(:,k));
        ccDt(:,k) = lambda(:,k).^-1 .* (rt(:,k) - t(:,k) .* ccDlambda(:,k));
    end

    %% Predictor-Corrector Search Direction
    Dx = affDx + ccDx;
    Da = affDa + ccDa;
    Dlambda = affDlambda + ccDlambda;
    Dp = affDp + ccDp;
    Dt = affDt + ccDt;
    Dbeta = affDbeta + ccDbeta;

    %% Calculate Predictor-Corrector Step Length
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


    %% Update Solution
    gamma = 1 - (1/(ipmIter+5)^2);
    %gamma = 0.99;

    x = x + (gamma * alpha * Dx);
    a = a + (gamma * alpha * Da);
    lambda = lambda + (gamma * alpha * Dlambda);
    p = p + (gamma * alpha * Dp);
    t = t + (gamma * alpha * Dt);
    beta = beta + (gamma * alpha * Dbeta);

    dualGap(ipmIter+1) = ( lambda(:)' * t(:) )/(nd*N);
    fprintf('End of iteration [%d] - DualGap: %3.8f -> %3.8f\n',ipmIter, dualGap(ipmIter), dualGap(ipmIter+1));
    ipmIter = ipmIter+1;
end

%% Analysis
figure
plot(0:1:N,[xBar x]', 'LineWidth', 2);
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
