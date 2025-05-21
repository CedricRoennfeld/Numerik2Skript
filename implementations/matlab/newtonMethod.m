function [x, fx, exitFlag] = newtonMethod(fun, jac, x0, maxIter, tol)
%NEWTON_METHOD Multidimensional Newton's method for root finding.
%   [X, FX, EXITFLAG] = NEWTON_METHOD(FUN, JAC, X0, MAXITER, TOL)
%   attempts to find a root of the function FUN using Newton's method,
%   starting from the initial guess X0. 
%
%   INPUTS:
%       fun      - Function handle for the function f(x), should accept
%                  and return a column vector.
%       jac      - Function handle for the Jacobian J(x) of f, should accept
%                  a column vector and return a matrix.
%       x0       - Initial guess (column vector).
%       maxIter  - Maximum number of iterations (default: 50).
%       tol      - Convergence tolerance (default: 1e-6).
%
%   OUTPUTS:
%       x        - Computed root (column vector).
%       fx       - Function value at the root (column vector).
%       exitFlag - Logical flag indicating if convergence was achieved.
%   EXAMPLE USAGE:
%       fun = @(x) [x(1)^2 + x(2)^2 - 1; x(1) - x(2)];
%       jac = @(x) [2*x(1), 2*x(2); 1, -1];
%       x0 = [0.5; 0.5];
%       x = newtonMethod(fun, jac, x0);

% default values
if nargin < 4 || isempty(maxIter), maxIter = 50; end
if nargin < 5 || isempty(tol), tol = 1e-6; end

x = x0(:); % Ensure column vector if x0 is a row vector
exitFlag = false;

for iter = 1:maxIter
    fx = fun(x);
    J = jac(x);
    dx = -J\fx; % solve J*dx = -fx
    x_new = x + dx; % Newton step
    % Check for convergence
    if norm(dx, inf) < tol
        exitFlag = true;
        return;
    end
    x = x_new;
end
end
