function [ x, iter ] = myFirstCGS( A, b, tol)
    x = b;
    r1 = b - A*x;
    d1 = r1;
    
    if norm(r1) < tol
        return
    end
    
    for iter = 1:length(b)
        alpha = (r1'*r1)/(d1'*(A*d1));
        x = x + alpha*d1;
        r2 = r1 - alpha*(A*d1);
        beta = (r2'*r2)/(r1'*r1);
        d2 = r2 + beta*d1;
        
        if( norm(r2) < tol )
            return;
        end
        
        r1 = r2; d1 = d2;
    end

end

