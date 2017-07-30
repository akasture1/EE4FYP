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

AP = 1000*((1/l:1/l:1).^0.5)';
IP = 40;
P = AP-IP;
% P = 10*((1/l:1/l:1).^1.1)';

n = 4;
N = 2;
step = 5;
tau = (0:step:step*N)';
xd = step*rand(n,1);


% All tasks have the same arrival time (k=0) and same deadline (k=N)
%na = ones(n,1); 
%nj = N*ones(n,1); 

%% IPM SOLVER SETUP
tol = 1e-3;
iter=0;
nl = n*l;
nd = nl + (n+1);

% determine start and end points for each sub-block e.g. B, D etc.
dim = [n, nl, n];
blockdim = sum(dim);

pos = zeros(3,2);   
pos(1,1) = 1;
pos(1,2) = dim(1);
for k = 2:size(dim,2)
    pos(k,1) = sum(dim(1:k-1)) + 1;
    pos(k,2) = pos(k,1) + dim(k) - 1;
end

%----------------------------------------------------------------%
% Choose feasible starting point
% using +ve lagrange multipliers (p, lambda) and slack variable (t)
x = zeros(n,N+1);
a = zeros(nl,N);
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

%% BEGIN
dgap = (lambda(:)' * t(:))/nd;
while dgap > tol
    
    %% LHS (JACOBIAN)
    LHS = [];
    for k = 1:N    
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd);
        R = -D' * invE * D;
      
        % prepare sub-blocks
        uBlock = sparse(sum(dim),sum(dim));
        
        %(1,3) I
        p1=1;p2=3;
        uBlock(pos(p1,1):pos(p1,2), pos(p2,1):pos(p2,2)) = speye(n);    

        %(2,2) R
        p1=2;p2=2;
        uBlock(pos(p1,1):pos(p1,2), pos(p2,1):pos(p2,2)) = R;    

        %(2,3) B'
        p1=2;p2=3;
        uBlock(pos(p1,1):pos(p1,2), pos(p2,1):pos(p2,2)) = B';    

        %(3,1) I
        p1=3;p2=1;
        uBlock(pos(p1,1):pos(p1,2), pos(p2,1):pos(p2,2)) = speye(n);

        %(3,2) B
        p1=3;p2=2;
        uBlock(pos(p1,1):pos(p1,2), pos(p2,1):pos(p2,2)) = B; 
        
        LHS = blkdiag(LHS,uBlock);
    end
    
    % resize to accommodate last few blocks
    [h,w] = size(LHS);
    [i,j,s] = find(LHS);
    
    % add the 2 missing -I matrices in position (4,3) and (3,4) that we skipped
    for k = 1:N-1
        p1 = (k*blockdim +1 :1: k*blockdim + n)';
        p2 = ((k-1)*blockdim + pos(3,1): 1 :(k-1)*blockdim +pos(3,2))';
        i = [i; p1; p2];
        j = [j; p2; p1];
        s = [s; -1*ones(2*n,1)];
    end
    
    % append 2 additional I blocks 
    k = N;
    p1 = (k*blockdim +1 :1: k*blockdim + 2*n)';
    p2 = ((k-1)*blockdim + pos(3,1): 1 :(k-1)*blockdim + pos(3,2) + n)';
    i = [i; p1; p2];
    j = [j; p2; p1];
    s = [s; repmat([-1*ones(n,1); ones(n,1)],2,1)];

    LHS = sparse(i,j,s,(h+2*n), (w+2*n));
    clear i j s

    %% PREDICTOR RHS
    
    rx = zeros(n,N+1);
    ra = zeros(nl,N);
    rlambda = zeros(nd,N);    
    rp = zeros(n,N);
    rt = zeros(nd,N);
    rbeta = zeros(n,1);
    ra_hat = zeros(nl,N);
    rlambda_hat = zeros(nd,N);
    
    
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
    
    % complete Predictor RHS
    rx(:,1) = -p(:,1);
    rp(:,1) = -(xd + B * a(:,1) - x(:, 2));
    for k = 2:N
        rx(:,k) = -(p(:,k) - p(:,k-1));
        rp(:,k) = -(x(:,k) + B * a(:,k) - x(:, k+1));

    end
    rx(:,N+1) = -(beta - p(:,N));
    rbeta = -x(:,N+1);

    cascadedRs = vertcat(rx(:,1:N), ra_hat, rp);
    PRHS = [cascadedRs(:); rx(:,N+1); rbeta];
    
    %% SOLVE PREDICTOR 
    PSOL = LHS \ PRHS;
    
    dx_aff = zeros(n,N+1);
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

        jump = (k-1)*blockdim;
        dx_aff(:,k) = PSOL(jump + 1 : jump + n );
        da_aff(:,k) = PSOL(jump + n + 1 : jump + n + nl);
        dp_aff(:,k) = PSOL(jump + n + nl + 1 : jump + n + nl + n );
        
        dlambda_aff(:,k) = invE * rlambda_hat(:,k) - invE * D * da_aff(:,k);
        dt_aff(:,k) = invLambda * rt(:,k) - invLambda * T * dlambda_aff(:,k);
    end
    dx_aff(:,N+1) = PSOL(N * blockdim + 1 : N * blockdim + n);
    dbeta_aff(:,1) = PSOL(N * blockdim + n + 1 : N * blockdim + 2*n);
      
    %% CORRECTOR RHS
    % find best affine step - % only need to consider the cases where dlambda is negative
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
      
    rx = zeros(n,N+1);
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
        invE = spdiags(-t(:,k).^-1 .* lambda(:,k), 0, nd , nd );
        
        rlambda_hat(:,k) = rlambda(:,k) - invLambda * rt(:,k);
        ra_hat(:,k) = ra(:,k) - D' * invE * rlambda_hat(:,k);    
    end  

    cascadedRs = vertcat(rx(:,1:N), ra_hat, rp);
    CRHS = [cascadedRs(:); rx(:,N+1); rbeta];
    
    %% SOLVE CORRECTOR 
    CSOL = LHS \ CRHS;
    
    dx_cc = zeros(n,N+1);
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
        
        jump = (k-1)*blockdim;
        dx_cc(:,k) = CSOL(jump + 1 : jump + n );
        da_cc(:,k) = CSOL(jump + n + 1 : jump + n + nl);
        dp_cc(:,k) = CSOL(jump + n + nl + 1 : jump + n + nl + n );

        dlambda_cc(:,k) = invE * rlambda_hat(:,k) - invE * D * da_cc(:,k);
        dt_cc(:,k) = invLambda * rt(:,k) - invLambda * T * dlambda_cc(:,k);
    end
    dx_cc(:,N+1) = CSOL(N * blockdim + 1 : N * blockdim + n);
    dbeta_cc(:,1) = CSOL(N * blockdim + n + 1 : N * blockdim + 2*n);
    
    %% DETERMINE SEARCH DIRECTION
    Dx = dx_aff + dx_cc;
    Da = da_aff + da_cc;
    Dlambda = dlambda_aff + dlambda_cc;
    Dp = dp_aff + dp_cc;
    Dt = dt_aff + dt_cc;
    Dbeta = dbeta_aff + dbeta_cc;
    
    % find best affine + centering step size
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
    gamma = 1-1/(iter+2)^2;
    
    x = x + gamma * alpha * Dx;
    a = a + gamma * alpha * Da;
    lambda = lambda + gamma * alpha * Dlambda;
    p = p + gamma * alpha * Dp;
    t = t + gamma * alpha * Dt;
    beta = beta + gamma * alpha * Dbeta;
    
    dgap = ( lambda(:)' * t(:) )/nd
    iter = iter+1;    
end  




