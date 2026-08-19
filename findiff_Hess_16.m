function [Hessfx] = findiff_Hess_16(x, h)
% Function that approximates the Hessian of the Banded trigonometric
% problem exploiting the diagonal structure

x = x(:); % column vector
n = length(x);

if isscalar(h)
    h = h * ones(n, 1);
end

x_plus  = x + h;
x_minus = x - h;
k_vec   = (1:n-1)';

diag_H = zeros(n, 1);

% k < n
f_base_n1  = k_vec .* (1 - cos(x(1:n-1))) + 2 * sin(x(1:n-1));
f_plus_n1  = k_vec .* (1 - cos(x_plus(1:n-1))) + 2 * sin(x_plus(1:n-1));
f_minus_n1 = k_vec .* (1 - cos(x_minus(1:n-1))) + 2 * sin(x_minus(1:n-1));

diag_H(1:n-1) = (f_plus_n1 - 2*f_base_n1 + f_minus_n1) ./ (h(1:n-1).^2);

% k = n
f_base_n  = n * (1 - cos(x(n))) - (n - 1) * sin(x(n));
f_plus_n  = n * (1 - cos(x_plus(n))) - (n - 1) * sin(x_plus(n));
f_minus_n = n * (1 - cos(x_minus(n))) - (n - 1) * sin(x_minus(n));

diag_H(n) = (f_plus_n - 2*f_base_n + f_minus_n) / (h(n)^2);

Hessfx = spdiags(diag_H, 0, n, n); %sparse
end