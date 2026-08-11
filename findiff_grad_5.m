function [gradfx] = findiff_grad_5(x, h, type)
% Function that approximates the gradient of the Generalized Broyden tridiagonal function exploiting the sparsity and the tridiagonal structure

    x = x(:);
    n = length(x);
    p = 7/3;
    gradfx = zeros(size(x));

    x_prev = [0; x(1:n-1)];
    x_next = [x(2:n); 0];
    f_base = (3 - 2*x).*x - x_prev - x_next + 1;
    
    if isscalar(h)
        h = h * ones(n, 1);
    end

    switch type
        case 'fw'
            for k = 1:n
                idx = max(1, k-1) : min(n, k+1);
                
                F_base_local = sum(abs(f_base(idx)).^p);
                
                hk = h(k);
                xh = x;
                xh(k) = xh(k) + hk;
                
                F_fw_local = 0;
                for i = idx
                    xi = xh(i);
                    xi_prev = 0; if i > 1, xi_prev = xh(i-1); end
                    xi_next = 0; if i < n, xi_next = xh(i+1); end
                    
                    fi_fw = (3 - 2*xi)*xi - xi_prev - xi_next + 1;
                    F_fw_local = F_fw_local + abs(fi_fw)^p;
                end
                
                gradfx(k) = (F_fw_local - F_base_local) / hk;
            end
            
        case 'c'
            for k = 1:n
                idx = max(1, k-1) : min(n, k+1);
                
                hk = h(k);
                xh_plus = x;  xh_plus(k) = xh_plus(k) + hk;
                xh_minus = x; xh_minus(k) = xh_minus(k) - hk;
                
                F_fw_local = 0;
                F_bw_local = 0;
                
                for i = idx
                    xi_fw = xh_plus(i);
                    xi_prev_fw = 0; if i > 1, xi_prev_fw = xh_plus(i-1); end
                    xi_next_fw = 0; if i < n, xi_next_fw = xh_plus(i+1); end
                    
                    fi_fw = (3 - 2*xi_fw)*xi_fw - xi_prev_fw - xi_next_fw + 1;
                    F_fw_local = F_fw_local + abs(fi_fw)^p;
                    
                    xi_bw = xh_minus(i);
                    xi_prev_bw = 0; if i > 1, xi_prev_bw = xh_minus(i-1); end
                    xi_next_bw = 0; if i < n, xi_next_bw = xh_minus(i+1); end
                    
                    fi_bw = (3 - 2*xi_bw)*xi_bw - xi_prev_bw - xi_next_bw + 1;
                    F_bw_local = F_bw_local + abs(fi_bw)^p;
                end
                
                gradfx(k) = (F_fw_local - F_bw_local) / (2 * hk);
            end
            
        otherwise
            error('Inserire un tipo valido: ''fw'' (forward) o ''c'' (centered)');
    end
end