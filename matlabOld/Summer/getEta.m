function Eta = getEta(n,N,l,Sigma)
    Eta = zeros(n,N);
    for k = 1:N
        for j = 1:n
            sel = 1+(j-1)*l:j*l;
            EInv = Sigma(n+1+sel,k).^-1;
            Eta(j,k) = Sigma(j,k)/(1 + Sigma(j,k)*sum(EInv));
        end
    end
end