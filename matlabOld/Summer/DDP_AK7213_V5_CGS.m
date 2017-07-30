function [aOpt, xOpt, ipmIter] = DDP_AK7213_V5_CGS(n, N, l, m, s, step, P, xBar, ipmTol, ipmMaxIter, solver, solverTol)

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
fprintf('Starting DDP_AK7213_V5_CGS for n=%d; N=%d; l=%d; m=%d\n',n,N,l,m);

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

s = -step*(1/l:1/l:1)';
B = kron(speye(n), s');

%% Start IPM iterations
dualGap(1) = (lambda(:)' * t(:))/(nd*N);
while dualGap(ipmIter) > ipmTol && ipmIter <= ipmMaxIter
    %fprintf('[%d] Duality Gap: %3.8f\n',ipmIter, dualGap(ipmIter));

   %% Compute Sigma, Eta, Rho
    Sigma = (t.^-1).*lambda;
    Eta = getEta(n,N,l,Sigma);
    Rho = getRho(n,N,l,Sigma,Eta);

	%% Compute LHS (Jacobian)
    U = zeros(n,N);
    V = zeros(n,N);
    for k = 1:N
        for j = 1+(k-1)*nj:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;

            %Term 1 - Diagonal
            U(j,k) = U(j,k) + sum((s.^2).*EInv) - Eta(j,k)*((s'*EInv).^2);

            %Term 2 - Rank 1 Matrix (v*v')
            V(j,k) = V(j,k) + s'*EInv - Eta(j,k)*sum(EInv)*(s'*EInv);
        end
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

	%% rx
	for k = 1:N-1
		sel = 1+(k-1)*nj:k*nj;
		bb = zeros(n,1); bb(sel) = beta(:,k);
		rx(:,k) = -bb - p(:,k+1) + p(:,k);
	end
	sel = 1+n-nj:n;
	bb = zeros(n,1); bb(sel) = beta(:,k);
	rx(:,N) = -bb + p(:,N) ;

    %% rbeta
    I = eye(n);
    for k = 1:N
		sel = 1+(k-1)*nj:k*nj;
        rbeta(:,k) = -xOpt(sel,k);
    end

    %% rp
    rp(:,1) = -xBar - B*aOpt(:,1) + xOpt(:,1);
    for k = 1:N-1
        rp(:,k+1) = -xOpt(:,k) - B*aOpt(:,k+1) + xOpt(:,k+1);
    end

    %% ra
    for k = 1:N
        ra(:,k) = -step*kron(ones(n,1),P) - B'*p(:,k) - D'*lambda(:,k);
    end

	%% rlambda, rt, rlambdaHat, raHat
	for k = 1:N
        rlambda(:,k) = -( D*aOpt(:,k) - d + t(:,k));
        rt(:,k) = -( lambda(:,k).*t(:,k) );

        rlambdaHat(:,k) = rlambda(:,k) - (lambda(:,k).^-1) .* rt(:,k);
        raHat(:,k) = ra(:,k) + D' * (t(:,k).^-1 .* lambda(:,k) .* rlambdaHat(:,k));
    end
    
    %% Compute PRHS
    PRHS = zeros(n,1);
    % Part 1: B*(R.^-1)*ra
    for k = 1:N
        u = zeros(n,1);
        v = zeros(n,1);      
        kap = 0;
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            u(j) = sum(s.*EInv.*raHat(sel,k)) - Eta(j,k)*(s'*EInv)*(EInv'*raHat(sel,k));
            v(j) = s'*EInv - Eta(j,k)*sum(EInv)*(s'*EInv);
            kap = kap + sum(EInv.*raHat(sel,k)) - Eta(j,k)*sum(EInv)*(EInv'*raHat(sel,k));
        end
        PRHS = PRHS + cropVector(u,nj,k) - Rho(k)*kap*cropVector(v,nj,k);
    end

    % Part 2: B*(R.^-1)*B'*sum(rx)
    for k = 1:N
        u = zeros(n,1);
        v = zeros(n,1);
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            u(j) = sum((s.^2).*EInv) - Eta(j,k)*((s'*EInv).^2);
            v(j) = s'*EInv - Eta(j,k)*sum(EInv)*(s'*EInv);
        end
        PRHS = PRHS + (cropVector(u,nj,k).*sum(rx(:,k:N),2)) - Rho(k)*(v'*sum(rx(:,k:N),2))*cropVector(v,nj,k);
    end

    % Part 3: rp
    for k = 1:N
        PRHS = PRHS - cropVector(rp(:,k),nj,k);
    end

    %Part 4: rbeta
    PRHS = PRHS - rbeta(:);

    %% Solve Predictor Problem using Custom CGS
    PSOL = cgsCustom(U,V,PRHS,Rho,solverTol);

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
	sel = 1+n-nj:n;
	bb = zeros(n,1); bb(sel) = affDbeta(:,k);
	affDp(:,N) = bb - rx(:,N);
	for k = N-1:-1:1
		sel = 1+(k-1)*nj:k*nj;
		bb = zeros(n,1); bb(sel) = affDbeta(:,k);
		affDp(:,k) = bb - rx(:,k) + affDp(:,k+1);
	end

	%% affDa
    for k = 1:N
        u = zeros(nl,1);
        v = zeros(nl,1);
        y = raHat(:,k) - B'*affDp(:,k);
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            u(sel) = (EInv.*y(sel)) - Eta(j,k)*EInv*(EInv'*y(sel));
            v(sel) = EInv - Eta(j,k)*EInv*(sum(EInv));            
        end
        kap = sum(u);
        affDa(:,k) = u - Rho(k)*kap*v;
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
    % Part 1: B*(R.^-1)*ra
    for k = 1:N
        u = zeros(n,1);
        v = zeros(n,1);      
        kap = 0;
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            u(j) = sum(s.*EInv.*raHat(sel,k)) - Eta(j,k)*(s'*EInv)*(EInv'*raHat(sel,k));
            v(j) = s'*EInv - Eta(j,k)*sum(EInv)*(s'*EInv);
            kap = kap + sum(EInv.*raHat(sel,k)) - Eta(j,k)*sum(EInv)*(EInv'*raHat(sel,k));
        end
        CRHS = CRHS + cropVector(u,nj,k) - Rho(k)*kap*cropVector(v,nj,k);
    end

    % Part 2: B*(R.^-1)*B'*sum(rx)
    for k = 1:N
        u = zeros(n,1);
        v = zeros(n,1);
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            u(j) = sum((s.^2).*EInv) - Eta(j,k)*((s'*EInv).^2);
            v(j) = s'*EInv - Eta(j,k)*sum(EInv)*(s'*EInv);
        end
        CRHS = CRHS + (cropVector(u,nj,k).*sum(rx(:,k:N),2)) - Rho(k)*(v'*sum(rx(:,k:N),2))*cropVector(v,nj,k);
    end

    % Part 3: rp
    for k = 1:N
        CRHS = CRHS - cropVector(rp(:,k),nj,k);
    end

    %Part 4: rbeta
    CRHS = CRHS - rbeta(:);
    %fprintf('[%d] Corrector RHS complete: %3.5fs\n',ipmIter, tcRHS(ipmIter));

	%% Solve Corrector Problem using Custom CGS
    CSOL = cgsCustom(U,V,CRHS,Rho,solverTol);

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
	sel = 1+n-nj:n;
	bb = zeros(n,1); bb(sel) = ccDbeta(:,k);
	ccDp(:,N) = bb - rx(:,N);
	for k = N-1:-1:1
		sel = 1+(k-1)*nj:k*nj;
		bb = zeros(n,1); bb(sel) = ccDbeta(:,k);
		ccDp(:,k) = bb - rx(:,k) + ccDp(:,k+1);
    end

    %% ccDa
    for k = 1:N
        u = zeros(nl,1);
        v = zeros(nl,1);
        y = raHat(:,k) - B'*ccDp(:,k);
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            u(sel) = (EInv.*y(sel)) - Eta(j,k)*EInv*(EInv'*y(sel));
            v(sel) = EInv - Eta(j,k)*EInv*(sum(EInv));            
        end
        kap = sum(u);
        ccDa(:,k) = u - Rho(k)*kap*v;
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