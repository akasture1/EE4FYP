function [ x, iter ] = cgsCustom(U,V,b,Mu,tol)
    N = size(U,2);
    x = b;
    
    % Compute F = A*x
    F = zeros(length(b),1);
    for k = 1:N
        F = F + (U(:,k).*x) - Mu(k)*(V(:,k)'*x)*V(:,k);
    end
    
    r1 = b - F; d1 = r1;
    if norm(r1) < tol
        return
    end
    
    for iter = 1:length(b)
        % Compute F = A*d1
        F = zeros(length(b),1);
        
        for k = 1:N
            F = F + (U(:,k).*d1) - Mu(k)*(V(:,k)'*d1)*V(:,k);
        end
        
        alpha = (r1'*r1)/(d1'*F);
        x = x + alpha*d1;
        r2 = r1 - alpha*F;
        beta = (r2'*r2)/(r1'*r1);
        d2 = r2 + beta*d1;
        
        if( norm(r2) < tol )
            return;
        end
        
        r1 = r2; d1 = d2;
    end
end