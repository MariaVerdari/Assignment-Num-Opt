function [Hessfx] = findiff_Hess_5(x, h)
% Function that approximates the Hesssian of the Generalized Broyden tridiagonal function exploiting the sparsity and the pentadiagonal structure

    x = x(:);
    n = length(x);
    p = 7/3;
    
    Hessfx = sparse(n, n);
    
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
        % --- 1. ELEMENTI SULLA DIAGONALE (j = k) ---
        % Se perturbiamo x_j, cambiano solo i termini f_{j-1}, f_j, f_{j+1}
        idx_diag = max(1, j-1) : min(n, j+1);
        
        xh_plus = x;  xh_plus(j) = xh_plus(j) + h;
        xh_minus = x; xh_minus(j) = xh_minus(j) - h;
        
        F_base  = F_local(x, idx_diag);
        F_plus  = F_local(xh_plus, idx_diag);
        F_minus = F_local(xh_minus, idx_diag);
        
        Hessfx(j,j) = (F_plus - 2*F_base + F_minus) / (h^2);
        
        % --- 2. ELEMENTI EXTRA-DIAGONALI (j = k+1, k+2) ---
        % Iteriamo solo fino alla seconda codiagonale (i = j+1, i = j+2)
        for i = j+1 : min(n, j+2)
            % Se perturbiamo x_i e x_j, il blocco di termini influenzati 
            % va da min(i,j)-1 a max(i,j)+1. Essendo i > j, va da j-1 a i+1.
            idx_off = max(1, j-1) : min(n, i+1);
            
            xh_ij = x; xh_ij(i) = xh_ij(i) + h; xh_ij(j) = xh_ij(j) + h;
            xh_i  = x; xh_i(i)  = xh_i(i) + h;
            xh_j  = x; xh_j(j)  = xh_j(j) + h;
            
            F_base_off = F_local(x, idx_off);
            F_ij       = F_local(xh_ij, idx_off);
            F_i        = F_local(xh_i, idx_off);
            F_j        = F_local(xh_j, idx_off);
            
            % Calcolo differenza finita mista
            Hessfx(i,j) = (F_ij - F_i - F_j + F_base_off) / (h^2);
            
            % Sfruttiamo il teorema di Schwarz (simmetria dell'Hessiana)
            Hessfx(j,i) = Hessfx(i,j);
        end
    end
end