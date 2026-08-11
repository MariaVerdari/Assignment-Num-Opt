function [gradfx] = findiff_grad_16(x, h, type)
% Function that approximates the gradient of the Banded trigonometric problem


    x = x(:);
    n = length(x);
    gradfx = zeros(size(x));

    switch type
        case 'fw'
            for k = 1:n
                xk = x(k);
                xk_h = x(k) + h;
                
                if k < n
                    f_base = k * (1 - cos(xk)) + 2 * sin(xk);
                    f_fw   = k * (1 - cos(xk_h)) + 2 * sin(xk_h);
                else 
                    f_base = n * (1 - cos(xk)) - (n - 1) * sin(xk);
                    f_fw   = n * (1 - cos(xk_h)) - (n - 1) * sin(xk_h);
                end
                
                gradfx(k) = (f_fw - f_base) / h;
            end
            
        case 'c'
            for k = 1:n
                xk_plus  = x(k) + h;
                xk_minus = x(k) - h;
                
                if k < n
                    f_plus  = k * (1 - cos(xk_plus)) + 2 * sin(xk_plus);
                    f_minus = k * (1 - cos(xk_minus)) + 2 * sin(xk_minus);
                else
                    f_plus  = n * (1 - cos(xk_plus)) - (n - 1) * sin(xk_plus);
                    f_minus = n * (1 - cos(xk_minus)) - (n - 1) * sin(xk_minus);
                end
                
                gradfx(k) = (f_plus - f_minus) / (2 * h);
            end
            
        otherwise
            error('Inserire un tipo valido: ''fw'' (forward) o ''c'' (centered)');
    end
end