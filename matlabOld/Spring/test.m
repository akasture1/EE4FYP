
%% Params
H = eye(n);
H1 = H(1:nj,:);
H2 = H(nj+1:2*nj,:);
H3 = H(end-nj+1:end,:);

R0 = D'*diag(t(:,1).^-1 .* lambda(:,1))*D;
R1 = D'*diag(t(:,2).^-1 .* lambda(:,2))*D;
R2 = D'*diag(t(:,3).^-1 .* lambda(:,3))*D;

invR0 = R0 \ eye(size(R0));
invR1 = R1 \ eye(size(R1));
invR2 = R2 \ eye(size(R2));

G0 = B*invR0*B';
G1 = B*invR1*B'; G1Bar = zeros(size(G1)); G1Bar(nj+1:end, nj+1:end) = G1(nj+1:end, nj+1:end);
G2 = B*invR1*B'; G2Bar = zeros(size(G2)); G2Bar(2*nj+1:end, 2*nj+1:end) = G1(2*nj+1:end, 2*nj+1:end);


%% LHS
AA = G0 + G1Bar + G2Bar;

%% More Params
J0 = B*invR0*raHat(:,1);
J1 = B*invR1*raHat(:,2); J1Bar = [zeros(nj,1);J1(nj+1:end)];
J2 = B*invR2*raHat(:,3); J2Bar = [zeros(2*nj,1);J2(2*nj+1:end)];

%% RHS
Term1 = rbeta(:);
Term2 = rp(:,1) + [zeros(nj,1);rp(nj+1:end,2)] + [zeros(2*nj,1);rp(2*nj+1:end,3)];
Term3 = J0 + J1Bar + J2Bar;
Term4 = G0*sum(rx(:,1:N),2) + G1*sum(rx(:,2:N),2) + G2*sum(rx(:,3:N),2);

bb = -Term1 - Term2 + Term3 + Term4;

% Solve!
SOLN = AA \bb;

%%
for j = 1:3
V(1:j,j+1) = 0
end

A1 = U(:,1)*V(:,1)' + U(:,2)*V(:,2)'+ U(:,3)*V(:,3)'+ U(:,4)*V(:,4)'