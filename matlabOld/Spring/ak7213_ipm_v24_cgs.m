%% ak7213_ipm_v24
% Matlab based Interior Point Method to solve a Linear Program
% v[2][4] indicates the following
% [2] - Same Arrival Time, Different Deadlines Problem
% [4] - Eliminating t, lambda, t, lambda, x, a, p

% Author: Anand Kasture (ak7213@ic.ac.uk)
% Date: 23 May 2017

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
% nj:    column vector containing number of deadlines at each step [nx1]

rng(2)
close all
clc
clear
profile on

% Problem Size
v2ProblemSize

fprintf('Starting IPM v24 for n=%d; N=%d; l=%d; m=%d\n',n,N,l,m);

%% IPM Solver Setup
nl = n * l;
nd = n + 1 + nl;

nj = ceil(n/N);
nr = zeros(N,1);            % k = 1,...,N
nr(1) = n;
for k = 1:N-1
    nr(k+1) = nr(k) - nj;
end

x = zeros(n,N);             % k = 1,...,N
a = zeros(nl,N);            % k = 0,...,N-1
lambda = ones(nd,N);        % k = 0,...,N-1
p = ones(n,N);              % k = 0,...,N-1
t = ones(nd,N);             % k = 0,...,N-1
beta = ones(nj,N);          % k = 1,...,N

% Debugging/Performance Flags
affAlphaVec = zeros(ipmMaxIter,1);
alphaVec = zeros(ipmMaxIter,1);
dualGap = zeros(ipmMaxIter+1,1);

method = getenv('method');

% Set up LHS components that do not change at each iteration
d = [ones(n,1); m; zeros(nl,1)];

D1 = kron(speye(n), ones(1,l));
D2 = ones(1,nl);
D3 = -1 * speye(nl);
D = [D1; D2; D3];

B = -kron(step * speye(n), s');

%% Start IPM iterations
%------------------------------------------------------------------------------%
dualGap(1) = (lambda(:)' * t(:))/(nd*N);
while dualGap(ipmIter) > ipmTol && ipmIter <= ipmMaxIter
    fprintf('[%d] Duality Gap: %3.8f\n',ipmIter, dualGap(ipmIter));

	%% Compute Inverse-Rs
	invMs = cell(N,1);
	Mu = zeros(N,1);
	for k = 1:N
		[invMs{k}, Mu(k)] = getRInvBlocks(lambda(:,k), t(:,k), n, l);
	end

	%% Compute LHS
	%---------------------------------------%
    U = zeros(n,N);
    V = zeros(n,N);
    for k = 1:N
		invM = invMs{k};
        U(:,k) = spdiags(B * invM * B',0);
        V(:,k) = B*sum(invM,2);
    end
    
	%% Initialise PRHS Variables
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
		HSelect = 1+(k-1)*nj:k*nj;
		bb = zeros(n,1); bb(HSelect) = beta(:,k);
		rx(:,k) = -bb - p(:,k+1) + p(:,k);
	end
	HSelect = 1+n-nj:n;
	bb = zeros(n,1); bb(HSelect) = beta(:,k);
	rx(:,N) = -bb + p(:,N) ;

    %% rbeta
    I = eye(n);
    for k = 1:N
		HSelect = 1+(k-1)*nj:k*nj;
        rbeta(:,k) = -x(HSelect,k);
    end

    %% rp
    rp(:,1) = -xBar - B*a(:,1) + x(:,1);
    for k = 1:N-1
        rp(:,k+1) = -x(:,k) - B*a(:,k+1) + x(:,k+1);
    end

    %% ra
    for k = 1:N
        ra(:,k) = -step*kron(ones(n,1),P) - B'*p(:,k) - D'*lambda(:,k);
    end

	%% rlambda, rt, rlambdaHat, raHat
	for k = 1:N
		rlambda(:,k) = -D*a(:,k) + d - t(:,k);
        rt(:,k) = -lambda(:,k).*t(:,k);

        invLambda = spdiags(lambda(:,k).^-1 , 0, nd , nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

        rlambdaHat(:,k) = rlambda(:,k) - invLambda * rt(:,k);
        raHat(:,k) = ra(:,k) - D' * invE * rlambdaHat(:,k);
	end

	%% Compute PRHS
	%---------------------------------------%
	PRHS = zeros(n,1);
	for k = 1:N
    invM = invMs{k};
    v = B*sum(invM,2);

    jj = B*(invM*raHat(:,k)) - Mu(k)*v*(sum(invM,1)*raHat(:,k));
    gg = B*(invM*(B'*sum(rx(:,k:N),2))) - Mu(k)*(v*(v'*sum(rx(:,k:N),2)));

		PRHS = PRHS - cropVector(rp(:,k),nj,k) ...
					+ cropVector(jj,nj,k) ...
					+ cropVector(gg,nj,k);
	end
	PRHS = PRHS - rbeta(:);

	%% Initialise Predictor Variables
    affDx = zeros(n,N);        % k = 1,...,N
    affDa = zeros(nl,N);       % k = 0,...,N-1
    affDp = zeros(n,N);        % k = 0,...,N-1
	affDbeta = zeros(nj,N);    % k = 1,...,N
	affDlambda = zeros(nd,N);  % k = 0,...,N-1
    affDt = zeros(nd,N);       % k = 0,...,N-1

	%% SOLVE PREDICTOR
    PSOL = ak7213_cgs(U,V,PRHS,Mu,solverTol);

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
    invM = invMs{k};
    v = sum(invM,2);
		affDa(:,k) = invM*(raHat(:,k) - B'*affDp(:,k)) ...
                 - Mu(k)*(v*(v'*(raHat(:,k) - B'*affDp(:,k))));
	end

	%% affDx
	affDx(:,1) = B*affDa(:,1) - rp(:,1);
	for k = 2:N
		affDx(:,k) = affDx(:,k-1) + B*affDa(:,k) - rp(:,k);
	end

	%% affDlambda, affDt
    for k = 1:N
        T = spdiags(t(:,k),0,nd,nd);
        invLambda = spdiags(lambda(:,k).^-1,0,nd,nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

        affDlambda(:,k) = (invE * rlambdaHat(:,k)) - (invE * D * affDa(:,k));
        affDt(:,k) = (invLambda * rt(:,k)) - (invLambda * T * affDlambda(:,k));
    end

    %% Find affine step length
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

    %% Calculate sigma = (affDualGap/dualGap)^3
    affDualGap = ( (lambda(:) + (affAlpha * affDlambda(:)))' * (t(:) + (affAlpha * affDt(:))) )/(nd*N);
    fprintf('[%d] Affine Duality Gap: %3.8f\n',ipmIter, affDualGap);

    sigma = (affDualGap/dualGap(ipmIter))^3;
    fprintf('[%d] Sigma: (%3.8f/%3.8f)^3 = %3.8f\n',ipmIter, affDualGap, dualGap(ipmIter),sigma);
    % Note: sigma ~ 0 represents good affine search direction

	%% Initialise CRHS Variables
    rx = zeros(n,N);            % k = 1,...,N
    ra = zeros(nl,N);           % k = 0,...,N-1
	rp = zeros(n,N);            % k = 0,...,N-1
    rbeta = zeros(nj,N);        % k = 1,...,N
    rlambda = zeros(nd,N);      % k = 0,...,N-1
    rt = -(affDlambda.*affDt) + (sigma * dualGap(ipmIter) * ones(nd, N));       % k = 0,...,N-1

	rlambdaHat = zeros(nd,N);
    raHat = zeros(nl,N);

    for k = 1:N
        invLambda = spdiags(lambda(:,k).^-1 , 0, nd , nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

		rlambdaHat(:,k) = rlambda(:,k) - (invLambda * rt(:,k));
        raHat(:,k) = ra(:,k) - (D' * invE * rlambdaHat(:,k));
    end

    %% Compute CRHS
	CRHS = zeros(n,1);
	for k = 1:N
    invM = invMs{k};
    v = B*sum(invM,2);

    jj = B*(invM*raHat(:,k)) - Mu(k)*v*(sum(invM,1)*raHat(:,k));
    gg = B*(invM*(B'*sum(rx(:,k:N),2))) - Mu(k)*(v*(v'*sum(rx(:,k:N),2)));

		CRHS = CRHS - cropVector(rp(:,k),nj,k) ...
					+ cropVector(jj,nj,k) ...
					+ cropVector(gg,nj,k);
	end
	CRHS = CRHS - rbeta(:);
    %fprintf('[%d] Corrector RHS complete: %3.5fs\n',ipmIter, tcRHS(ipmIter));

	%% Initialise Corrector Variables
    ccDx = zeros(n,N);        % k = 1,...,N
    ccDa = zeros(nl,N);       % k = 0,...,N-1
    ccDp = zeros(n,N);        % k = 0,...,N-1
	ccDbeta = zeros(nj,N);    % k = 1,...,N
	ccDlambda = zeros(nd,N);  % k = 0,...,N-1
    ccDt = zeros(nd,N);       % k = 0,...,N-1

	%% SOLVE CORRECTOR
    CSOL = ak7213_cgs(U,V,CRHS,Mu,solverTol);

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
    invM = invMs{k};
    v = sum(invM,2);
    ccDa(:,k) = invM*(raHat(:,k) - B'*ccDp(:,k)) ...
                 - Mu(k)*(v*(v'*(raHat(:,k) - B'*ccDp(:,k))));
	end

	%% ccDx
	ccDx(:,1) = B*ccDa(:,1) - rp(:,1);
	for k = 2:N
		ccDx(:,k) = ccDx(:,k-1) + B*ccDa(:,k) - rp(:,k);
	end

	%% ccDlambda, ccDt
    for k = 1:N
        T = spdiags(t(:,k),0,nd,nd);
        invLambda = spdiags(lambda(:,k).^-1,0,nd,nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

        ccDlambda(:,k) = (invE * rlambdaHat(:,k)) - (invE * D * ccDa(:,k));
        ccDt(:,k) = (invLambda * rt(:,k)) - (invLambda * T * ccDlambda(:,k));
    end

    %% Determine Search Direction
    Dx = affDx + ccDx;
    Da = affDa + ccDa;
    Dlambda = affDlambda + ccDlambda;
    Dp = affDp + ccDp;
    Dt = affDt + ccDt;
    Dbeta = affDbeta + ccDbeta;

    %% Find step length
    alpha = 1;
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

    gamma = 1 - (1/(ipmIter+5)^3);
    %gamma = 0.99;

    %% Update solution
    x = x + (gamma * alpha * Dx);
    a = a + (gamma * alpha * Da);
    lambda = lambda + (gamma * alpha * Dlambda);
    p = p + (gamma * alpha * Dp);
    t = t + (gamma * alpha * Dt);
    beta = beta + (gamma * alpha * Dbeta);

    dualGap(ipmIter+1) = ( lambda(:)' * t(:) )/(nd*N);
    fprintf('End of iteration [%d] - DualGap: %3.8f -> %3.8f\n',ipmIter, dualGap(ipmIter), dualGap(ipmIter+1));

    ipmIter = ipmIter+1;

%     figure
%     plot(0:1:N,[xBar, x]', 'LineWidth', 2)
%     title('Same-Arrival-Different-Deadline');
%     xlabel('Time Step [k]');
%     ylabel('Remaining Time (s)');
%     grid on
%     grid minor
end
  
%% Analysis
figure
plot(0:1:N,[xBar, x]', 'LineWidth', 2)
title('Same-Arrival-Different-Deadline');
xlabel('Time Step [k]');
ylabel('Remaining Time (s)');
grid on
grid minor

figure
semilogy(dualGap)
hold on
semilogy(1e-4);
title('Duality Gap Evolution');
xlabel('Time Step [k]');
ylabel('Duality Gap');
grid on
grid minor
% 
% figure
% X1 = kron(ones(1,n),[0:N]');
% Y1 = kron([1:n],ones(N+1,1));
% Z1 = [xBar, x]';
% plot3(X1,Y1,Z1, 'lineWidth', 2);
% title('Same-Arrival-Different-Deadline');
% xlabel('Time Step [k]');
% ylabel('Task');
% zlabel('Remaining Time (s)');
% grid on
% grid minor
% 
% figure
% cap = (sum(a)/m)';
% plot(cap);
profile report
