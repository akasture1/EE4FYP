function [ x, iter ] = cgsCustom2(U,V,b,Mu,solverTol,solverMaxIter)
    [n,N] = size(U);
    [x,~,~,iter] = cgs(@aFun, b, solverTol, solverMaxIter);
    
    function y = aFun(x)
        y = zeros(n,1);
        for k = 1:N
            y = y + (U(:,k).*x) - Mu(k)*(V(:,k)'*x)*V(:,k);
        end 
    end
end