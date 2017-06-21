function [ v ] = cropVector( v, nj, k )
    if k > 1
        s = nj*(k-1);
        v(1:s) = 0;
    end
end

