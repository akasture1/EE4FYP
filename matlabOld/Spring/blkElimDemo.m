n = 2;
m = 10;
p = 50;
q = 8;

A = rand(n,m);
B = rand(n,p);
C = rand(q,m);
D = rand(q,p);

x1 = rand(m,1);
x2 = rand(p,1);

rhs1 = [A B; C D] * [x1; x2];
rhs2 = [A*x1; C*x1] + [B*x2; D*x2];