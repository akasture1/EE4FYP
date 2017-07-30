close all
clear all
rng('default')

%% Mehrotra's Predictor-Corrector IPM

%% INPUT PARAMETER SETUP

% m:   scalar number of processors
% l:   scalar number of speed levels
% s:   column vector of speed levels [1xl]
% AP:  column vector of active power datas corresponding to each speed level [lx1]
% IP:  scalar constant for idle power data
% n:   scalar number of tasks
% N:   scalar number of discretization steps
% tau: column vector of time values [(N+1)x1]
% xd:  column vector containing minimum execution time of tasks [nx1]
% na:  column vector containing arrival time of tasks [nx1]
% nj:  column vector containing number of deadlines at each step [nx1]

% Simplified Dynamic Power Consumption Model: AP(s) = alpha * s^beta
% alpha > 0 ; beta >= 1

m = 100;
l = 10; 
s = (1/l:1/l:1)';

% AP = 1000*((1/l:1/l:1).^2)';
% IP = 40;
% P = AP-IP;
 P = 10*((1/l:1/l:1).^1.5)';

n = 60;
N = 20;
step = 5;
tau = (0:step:step*N)';
xd = step*rand(n,1);


% All tasks have the same arrival time (k=0) and same deadline (k=N)
%na = ones(n,1); 
%nj = N*ones(n,1); 

%% IPM SOLVER SETUP
tol = 1e-3;
iter= 0;
nl = n*l;
nd = n + 1 + nl;

% determine start and end points for each sub-block e.g. B, D etc.
dim = [n, nl, n];
blockdim = sum(dim);
fblockdim = blockdim - n;   % very first block 

%----------------------------------------------------------------%
% Choose feasible starting point
% using +ve lagrange multipliers (p, lambda) and slack variable (t)
x = zeros(n,N);			% k = 1,...,N        
a = zeros(nl,N);		% k = 0,...,N-1
lambda = ones(nd,N);    
p = ones(n,N);
t = ones(nd,N);  
beta = ones(n,1);

%----------------------------------------------------------------%
% Precalcualte matrices/vectors that remain constant over all iterates
d = [ones(n,1); m  ; zeros(nl,1)];

D1 = kron(speye(n), ones(1,l));
D2 = ones(1,nl);
D3 = -1 * speye(nl);
D = [D1;D2;D3];
clear D1 D2 D3

B = -kron(step * speye(n), s');


%% LHS (JACOBIAN)
[i,j,s] = find(B');
I = i + n;
J = j + n + nl;
S = s;

[i,j,s] = find(B);
I = [ I ; i + n + nl ];
J = [ J ; j + n ];
S = [ S ; s ];

[i,j,s] = find( speye(n) );
I = [ I ; i + n + nl; i ];
J = [ J ; j; j + n + nl ];
S = [ S ; s; s ];
    
%% BEGIN
dgap = (lambda(:)' * t(:))/nd;
while dgap > tol
    
    % Complete LHS (JACOBIAN)
    LHS = [];
    for k = 2:N    
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd);
        R = -D' * invE * D;
       
        [i,j,s] = find(R);
        ii = [ I; i + n ];
        jj = [ J; j + n ];
        ss = [ S; s ];

        LHS = blkdiag(LHS, sparse(ii,jj,ss, n+nl+n, n+nl+n));
        fprintf('completed LHS block %d\n', k);
    end
    
    % add very first block (this excludes the I blocks)
    invE = spdiags(-t(:,1).^-1 .* lambda(:,1), 0, nd , nd);
    R = -D' * invE * D;
    [ii,jj,ss] = find(R);
    
    [i,j,s] = find(B');
    ii = [ ii ; i ];
    jj = [ jj ; j + nl ];
    ss = [ ss ; s ];

    [i,j,s] = find(B);
    ii = [ ii ; i + nl ];
    jj = [ jj ; j ];
    ss = [ ss ; s ];
    
    LHS = blkdiag(sparse(ii,jj,ss, nl+n, nl+n), LHS);
    
    [h,w] = size(LHS);
    [ii,jj,ss] = find(LHS);
    
    % add the 2 missing -I matrices that we have thus far skipped 
    for k = 1:N-1
        [i,j,s] = find( -1*speye(n) );
        ii = [ ii ; i + fblockdim + k*blockdim; i + fblockdim + k*blockdim - n ];
        jj = [ jj ; j + fblockdim + k*blockdim - n; j + fblockdim + k*blockdim ];
        ss = [ ss ; s; s ];
    end

    % add the 2 missing -I matrices for the very first block 
    [i,j,s] = find( -1*speye(n) );
    ii = [ ii ; i + fblockdim; i + fblockdim - n ];
    jj = [ jj ; j + fblockdim - n; j + fblockdim ];
    ss = [ ss ; s; s ];
    
    % add the 2 missing -I and 2 missing I matrices at the very end
    [i,j,s] = find( speye(n) );
    ii = [ ii ; i + fblockdim + (N-1)*blockdim + n; i + fblockdim + (N-1)*blockdim ];
    jj = [ jj ; j + fblockdim + (N-1)*blockdim; j + fblockdim + (N-1)*blockdim + n ];
    ss = [ ss ; s; s ];

    LHS = sparse(ii,jj,ss,(h+2*n), (w+2*n));
    fprintf('completed LHS\n');
    
    %% PREDICTOR RHS
    
    rx = zeros(n,N);
    ra = zeros(nl,N);
    rlambda = zeros(nd,N);    
    rp = zeros(n,N);
    rt = zeros(nd,N); 
    rbeta = zeros(n,1);
    
	ra_hat = zeros(nl,N);
    rlambda_hat = zeros(nd,N);
    
    % ra
    for k = 1:N
        invLambda = spdiags(lambda(:,k).^-1 , 0, nd , nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );   
        
        ra(:,k) =  -(step * kron(ones(n,1), P) + B'*p(:,k) + D'*lambda(:,k));
        
        rlambda(:,k) = -(D * a(:,k) - d + t(:,k));
        rt(:,k) = -(lambda(:,k) .* t(:,k));
        % eliminating rt and rlambda
        rlambda_hat(:,k) = rlambda(:,k) - invLambda * rt(:,k);
        ra_hat(:,k) = ra(:,k) - D' * invE * rlambda_hat(:,k);    
    end  
    
    % rx
    for k = 1:N-1
        rx(:,k) = -(p(:,k+1) - p(:,k));
    end
    rx(:,N) = -(beta - p(:,N));
    
    % rp
    rp(:,1) = -(xd + B * a(:,1) - x(:, 1)); 
    for k = 2:N
        rp(:,k) = -(x(:,k-1) + B * a(:,k) - x(:, k));
    end
    
    cascadedRs = vertcat(rx(:,1:N-1), ra_hat(:,2:N), rp(:,2:N));
    PRHS = [ ra_hat(:,1); rp(:,1) ;cascadedRs(:); rx(:,N); rbeta];
    fprintf('completed Predictor RHS\n');
    
    %% SOLVE PREDICTOR   
    tic
    PSOL = minres(LHS, PRHS, 1e-6, 5000);
    %PSOL_M = minres(LHS, PRHS, 1e-4, 500);
    %PSOL = LHS \ PRHS;
    toc
    
    dx_aff = zeros(n,N);
    da_aff = zeros(nl,N);
    dlambda_aff = zeros(nd,N);    
    dp_aff = zeros(n,N);
    dt_aff = zeros(nd,N);
    dbeta_aff = zeros(n,1);
    
    % update predictor dx da and dp, and recover dlambda, dt from PSOL
    
	for k = 1:N
        T = spdiags(t(:,k),0,nd,nd);
        invLambda = spdiags(lambda(:,k).^-1,0,nd,nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

		if( k == 1 )
			da_aff(:,k) = PSOL( 1 : nl);
			dp_aff(:,k) = PSOL( nl + 1 : nl + n );
		else
			jump = nl + n + (k-2)*blockdim;
			
			dx_aff(:,k-1) = PSOL(jump + 1 : jump + n );
			da_aff(:,k) = PSOL(jump + n + 1 : jump + n + nl);
			dp_aff(:,k) = PSOL(jump + n + nl + 1 : jump + n + nl + n );
		end
		
        dlambda_aff(:,k) = invE * rlambda_hat(:,k) - invE * D * da_aff(:,k);
        dt_aff(:,k) = invLambda * rt(:,k) - invLambda * T * dlambda_aff(:,k);
    end
    
	dbeta_aff = PSOL( size(PSOL,1) - n + 1 : size(PSOL,1) );
      
    %% CORRECTOR RHS
    % find best affine step - % only need to consider the cases where dlambda_aff and dt_aff is negative
    alpha_aff = realmax;
    for i = 1:N
        for j = 1:nd
            if( dlambda_aff(j,i) < 0 )
                alpha_aff = min(alpha_aff, -lambda(j,i)/dlambda_aff(j,i));
            end
            if( dt_aff(j,i) < 0 )
                alpha_aff = min(alpha_aff, -t(j,i)/dt_aff(j,i));
            end
        end
    end
    
    % calculate sigma = (dgap_affine/dgap)^3
    dgap_aff = ( (lambda(:) + alpha_aff * dlambda_aff(:))' * (t(:) + alpha_aff * dt_aff(:)) )/nd;
    sigma = (dgap_aff/dgap)^3;
      
    rx = zeros(n,N);
    ra = zeros(nl,N);
    rlambda = zeros(nd,N);    
    rp = zeros(n,N);
    rt = -(dlambda_aff.*dt_aff) + sigma * dgap * ones(nd, N);
    rbeta = zeros(n,1);
    
    ra_hat = zeros(nl,N);
    rlambda_hat = zeros(nd,N);
    
    for k = 1:N   
        %eliminating rlambda and rt
        invLambda = spdiags(lambda(:,k).^-1,0,nd,nd);
        invE = spdiags( -t(:,k).^-1 .* lambda(:,k), 0, nd , nd );
        
        rlambda_hat(:,k) = rlambda(:,k) - invLambda * rt(:,k);
        ra_hat(:,k) = ra(:,k) - D' * invE * rlambda_hat(:,k);    
    end  
	
	cascadedRs = vertcat(rx(:,1:N-1), ra_hat(:,2:N), rp(:,2:N));
    CRHS = [ ra_hat(:,1); rp(:,1) ;cascadedRs(:); rx(:,N); rbeta];
    fprintf('completed Corrector RHS\n');

    %% SOLVE CORRECTOR
    
    tic
    CSOL = minres(LHS, CRHS, 1e-6, 5000);
    %CSOL_M = minres(LHS, CRHS, 1e-4, 500);
    %CSOL = LHS \ CRHS;
    toc
    
    dx_cc = zeros(n,N);
    da_cc = zeros(nl,N);
    dlambda_cc = zeros(nd,N);    
    dp_cc = zeros(n,N);
    dt_cc = zeros(nd,N);
    dbeta_cc = zeros(n,1);
    
    % update centering dx da and dp, and recover dlambda, dt
	for k = 1:N
        T = spdiags(t(:,k),0,nd,nd);
        invLambda = spdiags(lambda(:,k).^-1,0,nd,nd);
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );

		if( k == 1 )
			da_cc(:,k) = CSOL( 1 : nl);
			dp_cc(:,k) = CSOL( nl + 1 : nl + n );
		else
			jump = nl + n + (k-2)*blockdim;
			
			dx_cc(:,k-1) = CSOL(jump + 1 : jump + n );
			da_cc(:,k) = CSOL(jump + n + 1 : jump + n + nl);
			dp_cc(:,k) = CSOL(jump + n + nl + 1 : jump + n + nl + n );
		end
		
        dlambda_cc(:,k) = invE * rlambda_hat(:,k) - invE * D * da_cc(:,k);
        dt_cc(:,k) = invLambda * rt(:,k) - invLambda * T * dlambda_cc(:,k);
    end
    
	dbeta_cc = CSOL( size(CSOL,1) - n + 1 : size(CSOL,1) );
    
    %% DETERMINE SEARCH DIRECTION
    Dx = dx_aff + dx_cc;
    Da = da_aff + da_cc;
    Dlambda = dlambda_aff + dlambda_cc;
    Dp = dp_aff + dp_cc;
    Dt = dt_aff + dt_cc;
    Dbeta = dbeta_aff + dbeta_cc;
    
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
    
    %% UPDATE FOR NEXT ITERATION
    gamma = 1 - (1/(iter+5)^2);
    
    x = x + gamma * alpha * Dx;
    a = a + gamma * alpha * Da;
    lambda = lambda + gamma * alpha * Dlambda;
    p = p + gamma * alpha * Dp;
    t = t + gamma * alpha * Dt;
    beta = beta + gamma * alpha * Dbeta;

    dgap = ( lambda(:)' * t(:) )/nd
    iter = iter+1    
end  
