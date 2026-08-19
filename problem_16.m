function [F, gradF, HessF] = problem_16(x)

% Banded trigonometric problem

n = length(x);
i = (1:n)';

x_pad = [0; x; 0];

terms = i .* ( (1 - cos(x_pad(2:n+1))) + sin(x_pad(1:n)) - sin(x_pad(3:n+2)) );
F = sum(terms);

if nargout > 1 % if we don't want the gradient
    gradF = zeros(n, 1);
    
    gradF(1:n-1) = i(1:n-1) .* sin(x(1:n-1)) + 2 * cos(x(1:n-1));
    
    gradF(n) = n * sin(x(n)) - (n-1) * cos(x(n));
end

if nargout > 2 % if we don't want the Hessian
    d = zeros(n, 1);
    
    d(1:n-1) = i(1:n-1) .* cos(x(1:n-1)) - 2 * sin(x(1:n-1));
    
    d(n) = n * cos(x(n)) + (n-1) * sin(x(n));
    
    HessF = spdiags(d, 0, n, n);
end
end