function [Hessfx] = findiff_Hess_16(x, h)
% Function that approximates the Hessian of the Banded trigonometric problem

    x = x(:);
    n = length(x);
    
    Hessfx = sparse(n, n);
    
    if isscalar(h)
        h = h * ones(n, 1);
    end

    for k = 1:n
        xk = x(k);
        
        hk = h(k); 
        
        xk_plus  = xk + hk;
        xk_minus = xk - hk;
        

        if k < n
            f_base  = k * (1 - cos(xk)) + 2 * sin(xk);
            f_plus  = k * (1 - cos(xk_plus)) + 2 * sin(xk_plus);
            f_minus = k * (1 - cos(xk_minus)) + 2 * sin(xk_minus);
        else 
            f_base  = n * (1 - cos(xk)) - (n - 1) * sin(xk);
            f_plus  = n * (1 - cos(xk_plus)) - (n - 1) * sin(xk_plus);
            f_minus = n * (1 - cos(xk_minus)) - (n - 1) * sin(xk_minus);
        end
        
        Hessfx(k, k) = (f_plus - 2*f_base + f_minus) / (hk^2);
    end
end