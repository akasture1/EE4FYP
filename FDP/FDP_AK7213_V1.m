function [aOpt, xOpt, ipmIter] = FDP_AK7213_V1(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol)

%% Function Argument Definitions
% n:          scalar number of tasks
% N:          scalar number of discrete time steps
% l:          scalar number of speed levels
% m:          scalar number of processors
% s:          column vector of speed levels [1xl]
% step:       fixed time step between each point on the time grid
% P:          column vector of net power consumption values corresponding to each speed level [lx1]
% xBar:       column vector containing minimum execution time of tasks [nx1]
% ipmTol:     interior-point method stopping threshold
% ipmMaxIter: maximum number of interior-point method iterations
% solver:     system of linear equations solver
% solverTol:  used for iterative solvers like minres and cgs

%% IPM Solver Setup
fprintf('Starting FDP_AK7213_V1 for n=%d; N=%d; l=%d; m=%d\n',n,N,l,m);
nl = n * l;
nd = n + 1 + nl;

xOpt = step*rand(n,N);         % k = 1,...,N
aOpt = rand(nl,N);             % k = 0,...,N-1
lambda = ones(nd,N);        % k = 0,...,N-1
p = ones(n,N);              % k = 0,...,N-1
t = ones(nd,N);             % k = 0,...,N-1
beta = ones(n,1);           % k = 1,...,N

solverTol = 1e-6;
solverMaxIter = 20000;
ipmIter= 1;

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
rowOff = zeros(5,5);
colOff = zeros(5,5);
dim = [nl nd n nd n];
for i = 1:5
    for j = 1:5
        rowOff(i,j) = sum(dim(1:(i-1)));
        colOff(i,j) = sum(dim(1:(j-1)));
    end
end

% row 1
[i,j,s12] = find(D');
i12 = i + rowOff(1,2);
j12 = j + colOff(1,2);

[i,j,s13] = find(B');
i13 = i + rowOff(1,3);
j13 = j + colOff(1,3);

% row 2
[i,j,s21] = find(D);
i21 = i + rowOff(2,1);
j21 = j + colOff(2,1);

[i,j,s24] = find(speye(nd));
i24 = i + rowOff(2,4);
j24 = j + colOff(2,4);

 % row 3
[i,j,s31] = find(B);
i31 = i + rowOff(3,1);
j31 = j + colOff(3,1);

% Minus Is
[i,j,s] = find(-1*speye(n));
i53 = i + rowOff(5,3);
j53 = j + colOff(5,3);

i35 = i + rowOff(3,5);
j35 = j + colOff(3,5);

% set up T and Lambda indices for block 0
i = (1:1:nd)';
j = (1:1:nd)';
t0Rows = i + rowOff(4,2);
t0Cols = j + colOff(4,2);
l0Rows = i + rowOff(4,4);
l0Cols = j + colOff(4,4);

blk0Rows = [i12; i13; i21; i24; i31; i53; i35];
blk0Cols = [j12; j13; j21; j24; j31; j53; j35];
blk0Vals = [s12; s13; s21; s24; s31; s; s];
% spy(sparse(blk0Rows, blk0Cols, blk0Vals));

outerBlk0Dim = sum(dim);
innerBlk0Dim = outerBlk0Dim - n;

% Block k (k = 1,2,...,N-1)
rowOff = zeros(6,6);
colOff = zeros(6,6);
dim = [n nl nd n nd n];

for i = 1:6
    for j = 1:6
        rowOff(i,j) = sum(dim(1:(i-1)));
        colOff(i,j) = sum(dim(1:(j-1)));
    end
end

outerBlk1Dim = sum(dim);
innerBlk1Dim = outerBlk1Dim - n;

[i,j,s] = find(speye(n));
i14 = i + rowOff(1,4);
j14 = j + colOff(1,4);

i41 = i + rowOff(4,1);
j41 = j + colOff(4,1);

blk1Rows = [blk0Rows + n; i14; i41];
blk1Cols = [blk0Cols + n; j14; j41];
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

% Block N (k = N)
jump = max(offsets);
[i,j,s] = find(speye(n));
i56 = i + rowOff(6,6) + jump;
j56 = j + colOff(6,6) + n + jump;

i65 = i + rowOff(6,6) + n + jump;
j65 = j + colOff(6,6) + jump;

% Combine!
Rows = [blk0Rows; blkKRows; i56; i65];
Cols = [blk0Cols; blkKCols; j56; j65];
Vals = [blk0Vals; blkKVals; s; s];
%spy(sparse(Rows, Cols, Vals));

% Set up Row/Col Indices for all T and Lambda
i = (1:1:nd)';
j = (1:1:nd)';
tKRows = i + rowOff(5,3);
tKCols = j + colOff(5,3);
lKRows = i + rowOff(5,5);
lKCols = j + colOff(5,5);

mult = repmat((0:1:N-2), nd,1);
mult = mult(:);
offset1 = innerBlk0Dim * ones((N-1) * nd ,1);
offset2 = mult .* (innerBlk1Dim * ones((N-1) * nd ,1));
offsets = offset1 + offset2;

TLRows = [t0Rows; repmat(tKRows, N-1,1) + offsets; l0Rows; repmat(lKRows, N-1,1) + offsets];
TLCols = [t0Cols; repmat(tKCols, N-1,1) + offsets; l0Cols; repmat(lKCols, N-1,1) + offsets];
% TLVals change at each iteration of the IPM, so set within the while loop

%% Start IPM iterations
dualGap(1) = (lambda(:)' * t(:))/(nd*N);
while dualGap(ipmIter) > ipmTol && ipmIter <= ipmMaxIter
    %fprintf('[%d] Duality Gap: %3.8f\n',ipmIter, dualGap(ipmIter));

    % Construct Full LHS (JACOBIAN) with latest T and Lambda values
    LHS = sparse( [Rows; TLRows], [Cols; TLCols], [Vals; t(:); lambda(:)] );

    %% Construct Predictor RHS
    rx = zeros(n,N);        % k = 1,...,N
    ra = zeros(nl,N);       % k = 0,...,N-1
    rlambda = zeros(nd,N);  % k = 0,...,N-1
    rp = zeros(n,N);        % k = 0,...,N-1
    rt = zeros(nd,N);       % k = 0,...,N-1
    rbeta = zeros(n,1);

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
        rlambda(:,k) = -( D*aOpt(:,k) - d + t(:,k) );
    end

    % rp
    rp(:,1) = -( xBar + B*aOpt(:,1) - xOpt(:,1) );
    for k = 1:N-1
        rp(:,k+1) = -( xOpt(:,k) + B*aOpt(:,k+1) - xOpt(:,k+1) );
    end

    % rt
    for k = 1:N
        rt(:,k) = -( lambda(:,k).*t(:,k) );
    end

    % rbeta
    rbeta = -xOpt(:,N);

    blk0RHS = [ra(:,1); rlambda(:,1); rp(:,1); rt(:,1)];
    blkKRHS = [rx(:,1:N-1); ra(:,2:N); rlambda(:,2:N); rp(:,2:N); rt(:,2:N)];
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

	% dim = [n nl nd n nd n];
    blk0Sol = PSOL(1:innerBlk0Dim);
    blkKSol = reshape(PSOL(innerBlk0Dim+1:innerBlk0Dim + (N-1)*innerBlk1Dim),[innerBlk1Dim,(N-1)]);
    blkNSol = PSOL(end-(2*n)+1:end);

    affDa(:,1) = blk0Sol(1:dim(2));
    affDlambda(:,1) = blk0Sol(dim(2)+1:sum(dim(2:3)));
    affDp(:,1) = blk0Sol(sum(dim(2:3))+1:sum(dim(2:4)));
    affDt(:,1) = blk0Sol(sum(dim(2:4))+1:sum(dim(2:5)));

    affDx(:,1:N-1) = blkKSol(1:dim(1),:);
    affDa(:,2:N) = blkKSol(dim(1)+1:sum(dim(1:2)),:);
    affDlambda(:,2:N) = blkKSol(sum(dim(1:2))+1:sum(dim(1:3)),:);
    affDp(:,2:N) = blkKSol(sum(dim(1:3))+1:sum(dim(1:4)),:);
    affDt(:,2:N) = blkKSol(sum(dim(1:4))+1:sum(dim(1:5)),:);

    affDx(:,N) = blkNSol(1:dim(1));
    affDbeta = blkNSol(dim(1)+1:end);

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
    %fprintf('[%d] Affine Step Length: %3.8f\n',ipmIter, affAlpha);

    %% Calculate Sigma
    % sigma ~ 0 indicates that the affine dir. is a good search dir.
    affDualGap = ( (lambda(:) + (affAlpha * affDlambda(:)))' * (t(:) + (affAlpha * affDt(:))) )/(nd*N);
    %fprintf('[%d] Affine Duality Gap: %3.8f\n',ipmIter, affDualGap);

    sigma = (affDualGap/dualGap(ipmIter))^3;
    sigmaVec(ipmIter) = sigma;
    %fprintf('[%d] Sigma: (%3.8f/%3.8f)^3 = %3.8f\n',ipmIter, affDualGap, dualGap(ipmIter),sigma);

    %% Construct Corrector RHS
    rx = zeros(n,N);            % k = 1,...,N
    ra = zeros(nl,N);           % k = 0,...,N-1
    rlambda = zeros(nd,N);      % k = 0,...,N-1
    rp = zeros(n,N);            % k = 0,...,N-1
    rt = -(affDlambda.*affDt) + (sigma * dualGap(ipmIter) * ones(nd, N));       % k = 0,...,N-1
    rbeta = zeros(n,1);        % k = 1,...,N

    blk0RHS = [ra(:,1); rlambda(:,1); rp(:,1); rt(:,1)];
    blkKRHS = [rx(:,1:N-1); ra(:,2:N); rlambda(:,2:N); rp(:,2:N); rt(:,2:N)];
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

    ccDa(:,1) = blk0Sol(1:dim(2));
    ccDlambda(:,1) = blk0Sol(dim(2)+1:sum(dim(2:3)));
    ccDp(:,1) = blk0Sol(sum(dim(2:3))+1:sum(dim(2:4)));
    ccDt(:,1) = blk0Sol(sum(dim(2:4))+1:sum(dim(2:5)));

    ccDx(:,1:N-1) = blkKSol(1:dim(1),:);
    ccDa(:,2:N) = blkKSol(dim(1)+1:sum(dim(1:2)),:);
    ccDlambda(:,2:N) = blkKSol(sum(dim(1:2))+1:sum(dim(1:3)),:);
    ccDp(:,2:N) = blkKSol(sum(dim(1:3))+1:sum(dim(1:4)),:);
    ccDt(:,2:N) = blkKSol(sum(dim(1:4))+1:sum(dim(1:5)),:);

    ccDx(:,N) = blkNSol(1:dim(1));
    ccDbeta = blkNSol(dim(1)+1:end);

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
    %fprintf('[%d] Alpha (Step Length): %3.8f\n',ipmIter, alpha);

    %% Update Solution
    %gamma = 1 - (1/(ipmIter+5)^2.5);
    gamma = 0.99;

    xOpt = xOpt + (gamma * alpha * Dx);
    aOpt = aOpt + (gamma * alpha * Da);
    lambda = lambda + (gamma * alpha * Dlambda);
    p = p + (gamma * alpha * Dp);
    t = t + (gamma * alpha * Dt);
    beta = beta + (gamma * alpha * Dbeta);

    dualGap(ipmIter+1) = ( lambda(:)' * t(:) )/(nd*N);
    %fprintf('End of iteration [%d] - DualGap: %3.8f --> %3.8f\n',ipmIter, dualGap(ipmIter), dualGap(ipmIter+1));
    ipmIter = ipmIter+1;
end

fprintf('Completed FDP_AK7213_V1 for n=%d; N=%d; l=%d; m=%d\n',n,N,l,m);
end
