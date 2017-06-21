function [ A ] = cropMatrix( A, nj, k )
    if k > 1
        s = nj*(k-1);
        [r,c,v] = find(A);
        v((c<=s)|(r<=s)) = 0;
        A = sparse(r,c,v);
%         A(1:s,:) = 0;
%         A(:,1:s) = 0;
    end
end