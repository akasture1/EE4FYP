function [ MInv, mu ] = getRInvBlocks( lambda, t, n, l )
	Sigma = (t.^-1).*lambda;
	EBar = Sigma(n+2:end);
	MInvs = cell(n,1);
    for j = 1:n
        EInv = EBar(1+(j-1)*l:j*l).^-1;
        beta = Sigma(j)/(1 + Sigma(j)*sum(EInv));
        MInvs{j} = sparse(diag(EInv) - beta*(EInv*EInv'));
    end
    MInv = blkdiag(MInvs{:});
    mu = Sigma(n+1)/(1 + Sigma(n+1)*sum(sum(MInv)));
end
