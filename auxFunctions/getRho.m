function Rho = getRho(n,N,l,Sigma,Beta)
    Rho = zeros(1,N);
    for k = 1:N
        v = zeros(n*l,1);
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            v(sel) = EInv - Beta(j,k)*sum(EInv)*EInv;
        end
        Rho(k) = Sigma(n+1,k)/(1+Sigma(n+1,k)*sum(v));
    end
end