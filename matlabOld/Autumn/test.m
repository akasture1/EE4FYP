% want to verify the application
% of the sherman morrison formula
close all

l = 10;
s = (0.1:0.1:l*0.1)';

nr = 10;
nr_next = 8;
nj = 2;

B = kron(eye(nr_next),s');
figure; spy(B); title('B');

D1 = kron(eye(nr_next),ones(1,l)); [r1,c] = size(D1);
D2 = ones(1,nr_next*l);            [r2,~] = size(D2);
D3 = -eye(nr_next*l);              [r3,~] = size(D3);
D = [ D1 ; D2 ; D3  ];
figure; spy(D); title('D');

H = [eye(nj), zeros(nj,nr-nj) ];
E = [zeros(nr_next,nj), eye(nr_next) ];

nd = size(D,1);
t = rand(nd,1);
lambda = rand(nd,1);
invE = diag(-t.^-1 .* lambda, 0);
R = -D'*invE*D; figure; spy(R); title('R');

R1 = - [D1; zeros(r2+r3,c)]'*invE*[D1; zeros(r2+r3,c)];
R2 = - [zeros(r1,c);D2;zeros(r3,c3)]'*invE*[zeros(r1,c);D2;zeros(r3,c)];
R3 = - [zeros(r1+r2,c);D3]'*invE*[zeros(r1+r2,c);D3];

figure; spy(R1); title('R1');
figure; spy(R2); title('R2');
figure; spy(R3); title('R3');

