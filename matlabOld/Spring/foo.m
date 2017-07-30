x = rand(1e5,1);
a = ones(1,1e5);
tic
sum(x);
ts = toc

tic
a*x;
ts = toc