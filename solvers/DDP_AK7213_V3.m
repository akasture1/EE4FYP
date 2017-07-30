function [aOpt, xOpt, ipmIter] = DDP_AK7213_V3(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol, solverMaxIter)

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
fprintf('Starting DDP_AK7213_V3 for n=%d; N=%d; l=%d; m=%d\n',n,N,l,m);

nl = n * l;
nd = n + 1 + nl;
ipmIter = 1;

nj = ceil(n/N);
nr = zeros(N,1);            % k = 1,...,N
nr(1) = n;
for k = 1:N-1
    nr(k+1) = nr(k) - nj;
end

xOpt = step*rand(n,N);      % k = 1,...,N
aOpt = rand(nl,N);          % k = 0,...,N-1
lambda = ones(nd,N);        % k = 0,...,N-1
p = ones(n,N);              % k = 0,...,N-1
t = ones(nd,N);             % k = 0,...,N-1
beta = ones(nj,N);          % k = 1,...,N

%% Debugging/Performance Flags
alphaVec = zeros(ipmMaxIter,1);
sigmaVec = zeros(ipmMaxIter,1);
dualGap = zeros(ipmMaxIter+1,1);

%% Constant LHS Componentsd = [ones(n,1); m; zeros(nl,1)];
d = [ones(n,1); m; zeros(nl,1)];

D1 = kron(speye(n), ones(1,l));
D2 = ones(1,nl);
D3 = -1 * speye(nl);
D = [D1; D2; D3];

B = -kron(step * speye(n), s');

%% Start IPM iterations
dualGap(1) = (lambda(:)' * t(:))/(nd*N);
while dualGap(ipmIter) > ipmTol && ipmIter <= ipmMaxIter
    %fprintf('[%d] Duality Gap: %3.8f\n',ipmIter, dualGap(ipmIter));

    %% Compute Inverse-Rs
    invRs = zeros(nl,nl,N);
    for k = 1:N
        R = D'*diag(t(:,k).^-1 .* lambda(:,k))*D;
        invRs(:,:,k) = R \ eye(size(R));
    end

	%% Compute LHS (Jacobian)
	LHS = zeros(n,n);
	for k = 1:N
		G = B * invRs(:,:,k) * B';
		LHS = LHS + cropMatrix(G,nj,k);
	end

    %% Construct Predictor RHS
	rx = zeros(n,N);        	% k = 1,...,N
    ra = zeros(nl,N);       	% k = 0,...,N-1
    rp = zeros(n,N);        	% k = 0,...,N-1
	rbeta = zeros(nj,N);    	% k = 1,...,N
	rlambda = zeros(nd,N);  	% k = 0,...,N-1
    rt = zeros(nd,N);       	% k = 0,...,N-1

	rlambdaHat = zeros(nd,N);
    raHat = zeros(nl,N);

	% rx
	for k = 1:N-1
		HSelect = 1+(k-1)*nj:k*nj;
		bb = zeros(n,1); bb(HSelect) = beta(:,k);
		rx(:,k) = -bb - p(:,k+1) + p(:,k);
	end
	HSelect = 1+n-nj:n;
	bb = zeros(n,1); bb(HSelect) = beta(:,k);
	rx(:,N) = -bb + p(:,N) ;

    % rbeta
    I = eye(n);
    for k = 1:N
		HSelect = 1+(k-1)*nj:k*nj;
        rbeta(:,k) = -xOpt(HSelect,k);
    end

    % rp
    rp(:,1) = -xBar - B*aOpt(:,1) + xOpt(:,1);
    for k = 1:N-1
        rp(:,k+1) = -xOpt(:,k) - B*aOpt(:,k+1) + xOpt(:,k+1);
    end

    % ra
    for k = 1:N
        ra(:,k) = -step*kron(ones(n,1),P) - B'*p(:,k) - D'*lambda(:,k);
    end

	% rlambda, rt, rlambdaHat, raHat
	for k = 1:N
        rlambda(:,k) = -( D*aOpt(:,k) - d + t(:,k));
        rt(:,k) = -( lambda(:,k).*t(:,k) );

        rlambdaHat(:,k) = rlambda(:,k) - (lambda(:,k).^-1) .* rt(:,k);
        raHat(:,k) = ra(:,k) + D' * (t(:,k).^-1 .* lambda(:,k) .* rlambdaHat(:,k));
	end

	%% Compute PRHS
	PRHS = zeros(n,1);
	for k = 1:N
		jj = B*(invRs(:,:,k)*raHat(:,k));
		gg = (B*invRs(:,:,k)) * (B'*sum(rx(:,k:N),2));
		PRHS = PRHS - cropVector(rp(:,k),nj,k) ...
					+ cropVector(jj,nj,k) ...
					+ cropVector(gg,nj,k);
	end
	PRHS = PRHS - rbeta(:);

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
    affDbeta = zeros(nj,N);    % k = 1,...,N

	%% affDbeta
	affDbeta = reshape(PSOL, [nj,N]);

	%% affDp
	HSelect = 1+n-nj:n;
	bb = zeros(n,1); bb(HSelect) = affDbeta(:,k);
	affDp(:,N) = bb - rx(:,N);
	for k = N-1:-1:1
		HSelect = 1+(k-1)*nj:k*nj;
		bb = zeros(n,1); bb(HSelect) = affDbeta(:,k);
		affDp(:,k) = bb - rx(:,k) + affDp(:,k+1);
	end

	%% affDa
	for k = 1:N
		affDa(:,k) = invRs(:,:,k)*(raHat(:,k) - B'*affDp(:,k));
	end

	%% affDx
	affDx(:,1) = B*affDa(:,1) - rp(:,1);
	for k = 2:N
		affDx(:,k) = affDx(:,k-1) + B*affDa(:,k) - rp(:,k);
	end

	%% affDlambda, affDt
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
    %fprintf('[%d] Affine Alpha (Step Length): %3.8f\n',ipmIter, affAlpha);

    %% Calculate Sigma
    % sigma ~ 0 indicates that the affine dir. is a good search dir.
    affDualGap = ( (lambda(:) + (affAlpha * affDlambda(:)))' * (t(:) + (affAlpha * affDt(:))) )/(nd*N);
    % fprintf('[%d] Affine Duality Gap: %3.8f\n',ipmIter, affDualGap);

    sigma = (affDualGap/dualGap(ipmIter))^3;
    sigmaVec(ipmIter) = sigma;
    %fprintf('[%d] Sigma: (%3.8f/%3.8f)^3 = %3.8f\n',ipmIter, affDualGap, dualGap(ipmIter),sigma);

    %% Construct Corrector RHS
    rx = zeros(n,N);            % k = 1,...,N
    ra = zeros(nl,N);           % k = 0,...,N-1
    rlambda = zeros(nd,N);      % k = 0,...,N-1
    rp = zeros(n,N);            % k = 0,...,N-1
    rt = -(affDlambda.*affDt) + (sigma * dualGap(ipmIter) * ones(nd, N));       % k = 0,...,N-1
    rbeta = zeros(nj,N);        % k = 1,...,N

    rlambdaHat = zeros(nd,N);
    raHat = zeros(nl,N);

    for k = 1:N
      rlambdaHat(:,k) = rlambda(:,k) - (lambda(:,k).^-1) .* rt(:,k);
      raHat(:,k) = ra(:,k) + D' * (t(:,k).^-1 .* lambda(:,k) .* rlambdaHat(:,k));
    end

    %% Compute CRHS
	CRHS = zeros(n,1);
	for k = 1:N
		jj = B*(invRs(:,:,k)*raHat(:,k));
		gg = (B*invRs(:,:,k)) * (B'*sum(rx(:,k:N),2));
		CRHS = CRHS - cropVector(rp(:,k),nj,k) ...
					+ cropVector(jj,nj,k) ...
					+ cropVector(gg,nj,k);
	end
	CRHS = CRHS - rbeta(:);

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
    ccDbeta = zeros(nj,N);    % k = 1,...,N

	%% ccDbeta
	ccDbeta = reshape(CSOL, [nj,N]);

	%% ccDp
	HSelect = 1+n-nj:n;
	bb = zeros(n,1); bb(HSelect) = ccDbeta(:,k);
	ccDp(:,N) = bb - rx(:,N);
	for k = N-1:-1:1
		HSelect = 1+(k-1)*nj:k*nj;
		bb = zeros(n,1); bb(HSelect) = ccDbeta(:,k);
		ccDp(:,k) = bb - rx(:,k) + ccDp(:,k+1);
	end

	%% ccDa
	for k = 1:N
		ccDa(:,k) = invRs(:,:,k)*(raHat(:,k) - B'*ccDp(:,k));
	end

	%% ccDx
	ccDx(:,1) = B*ccDa(:,1) - rp(:,1);
	for k = 2:N
		ccDx(:,k) = ccDx(:,k-1) + B*ccDa(:,k) - rp(:,k);
	end

	%% ccDlambda, ccDt
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
    %fprintf('[%d] Alpha (Step Length): %3.8f\n',ipmIter, alpha);

    %% Update Solution
    gamma = 1 - (1/(ipmIter+5)^2);
    %gamma = 0.99;

    xOpt = xOpt + (gamma * alpha * Dx);
    aOpt = aOpt + (gamma * alpha * Da);
    lambda = lambda + (gamma * alpha * Dlambda);
    p = p + (gamma * alpha * Dp);
    t = t + (gamma * alpha * Dt);
    beta = beta + (gamma * alpha * Dbeta);

    dualGap(ipmIter+1) = ( lambda(:)' * t(:) )/(nd*N);
    %fprintf('End of iteration [%d] - DualGap: %3.8f -> %3.8f\n',ipmIter, dualGap(ipmIter), dualGap(ipmIter+1));
    ipmIter = ipmIter+1;
end
if dualGap(ipmIter) > ipmTol 
    fprintf('Failed to find Solution\n')
end
end
