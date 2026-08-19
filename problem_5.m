function [F, gradF, HessF] = problem_5(x)

    % Generalized Broyden tridiagonal function

    x = x(:);
    n = length(x);
    p = 7/3;
    
    x_prev = [0; x(1:n-1)];
    x_next = [x(2:n); 0];
    
    f_val = (3 - 2*x).*x - x_prev - x_next + 1;
    
    F = sum(abs(f_val).^p);
    
    if nargout < 2 % if we don't want the gradient
        return;
    end
    
    g = p * (abs(f_val).^(p-1)) .* sign(f_val);
    g_prev = [0; g(1:n-1)];
    g_next = [g(2:n); 0];
    
    gradF = g .* (3 - 4*x) - g_prev - g_next;
    
    if nargout < 3  % if we don't want the Hessian
        return;
    end
    
    W = p*(p-1) * abs(f_val).^(p-2);
    W_prev = [0; W(1:n-1)];
    W_next = [W(2:n); 0];
    
    d0 = W .* (3 - 4*x).^2 - 4*g + W_prev + W_next;
    
    V = W .* (3 - 4*x);
    d1 = - V(1:n-1) - V(2:n);
    
    d2 = W(2:n-1);
    
    HessF = diag(sparse(d0), 0) + ...
            diag(sparse(d1), 1) + diag(sparse(d1), -1) + ...
            diag(sparse(d2), 2) + diag(sparse(d2), -2);
end