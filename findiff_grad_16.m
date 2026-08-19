function [gradfx] = findiff_grad_16(x, h, type)
% Function that approximates the gradient of the Banded trigonometric problem
x = x(:); % column vector
n = length(x);

if isscalar(h)
    h = h * ones(n, 1);
end

k_vec = (1:n-1)';

switch type
    case 'fw'
        x_h = x + h;
        
        gradfx = zeros(n, 1);
        
        % k < n
        f_base_n1 = k_vec .* (1 - cos(x(1:n-1))) + 2 * sin(x(1:n-1));
        f_fw_n1   = k_vec .* (1 - cos(x_h(1:n-1))) + 2 * sin(x_h(1:n-1));
        gradfx(1:n-1) = (f_fw_n1 - f_base_n1) ./ h(1:n-1);
        
        % k = n
        f_base_n = n * (1 - cos(x(n))) - (n - 1) * sin(x(n));
        f_fw_n   = n * (1 - cos(x_h(n))) - (n - 1) * sin(x_h(n));
        gradfx(n) = (f_fw_n - f_base_n) / h(n);
        
    case 'c'
        x_plus  = x + h;
        x_minus = x - h;
        
        gradfx = zeros(n, 1);
        
        % k < n
        f_plus_n1  = k_vec .* (1 - cos(x_plus(1:n-1))) + 2 * sin(x_plus(1:n-1));
        f_minus_n1 = k_vec .* (1 - cos(x_minus(1:n-1))) + 2 * sin(x_minus(1:n-1));
        gradfx(1:n-1) = (f_plus_n1 - f_minus_n1) ./ (2 * h(1:n-1));
        
        %  k = n
        f_plus_n  = n * (1 - cos(x_plus(n))) - (n - 1) * sin(x_plus(n));
        f_minus_n = n * (1 - cos(x_minus(n))) - (n - 1) * sin(x_minus(n));
        gradfx(n) = (f_plus_n - f_minus_n) / (2 * h(n));
        
    otherwise
        error('Inserted Type is not valid');
end
end