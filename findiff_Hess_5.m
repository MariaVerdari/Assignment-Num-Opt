function [Hessfx] = findiff_Hess_5(x, h)
% Function that approximates the Hesssian of the Generalized Broyden tridiagonal function exploiting the sparsity and the pentadiagonal structure

x = x(:); %column vector
n = length(x);
p = 7/3;

Hessfx = sparse(n, n);
if isscalar(h)
    h = h * ones(n, 1);
end

function val = F_local(xt, indices)
    val = 0;
    for m = indices
        xm = xt(m);
        xm_prev = 0; if m > 1, xm_prev = xt(m-1); end
        xm_next = 0; if m < n, xm_next = xt(m+1); end
        fm = (3 - 2*xm)*xm - xm_prev - xm_next + 1;
        val = val + abs(fm)^p;
    end
end

for j = 1:n
    idx_diag = max(1, j-1) : min(n, j+1);
    
    hj = h(j);
    xh_plus = x;  
    xh_plus(j) = xh_plus(j) + hj; % x + h * e_j (perturbed x)
    xh_minus = x; 
    xh_minus(j) = xh_minus(j) - hj; % x - h * e_j (perturbed x)
    
    F_base  = F_local(x, idx_diag);
    F_plus  = F_local(xh_plus, idx_diag);
    F_minus = F_local(xh_minus, idx_diag);
    
    Hessfx(j,j) = (F_plus - 2*F_base + F_minus) / (hj^2); %diagonal terms
    
    for i = j+1 : min(n, j+2)
        idx_off = max(1, j-1) : min(n, i+1);
        
        hi = h(i);
        xh_ij = x; 
        xh_ij(i) = xh_ij(i) + hi; % x + h * e_i (perturbed x)
        xh_ij(j) = xh_ij(j) + hj; % x + h * e_j (perturbed x)
        xh_i  = x;
        xh_i(i)  = xh_i(i) + hi; % x + h * e_i (perturbed x)
        xh_j  = x; 
        xh_j(j)  = xh_j(j) + hj; % x + h * e_j (perturbed x)
        
        F_base_off = F_local(x, idx_off);
        F_ij       = F_local(xh_ij, idx_off);
        F_i        = F_local(xh_i, idx_off);
        F_j        = F_local(xh_j, idx_off);
        
        Hessfx(i,j) = (F_ij - F_i - F_j + F_base_off) / (hi*hj); %non-diagonal terms
        
        Hessfx(j,i) = Hessfx(i,j); %simmetry
    end
end
end