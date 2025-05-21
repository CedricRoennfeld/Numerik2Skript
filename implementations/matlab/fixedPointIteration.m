function [x, iter, exitFlag] = fixedPointIteration(fun, x0, maxIter, tol)
%FIXEDPOINTITERATION    Fixed-point iteration for multidimensional functions.
%   [X, ITER, EXITFLAG] = FIXEDPOINTITERATION(FUN, X0, TOL, MAXITER)
%   attempts to solve the fixed point equation X = F(X) using fixed point 
%   iteration starting from initial guess X0.
%
%   INPUTS:
%       fun      - Function handle for the iteration function f(x), 
%                  should accept and return a column vector.
%       x0       - Initial guess (column vector).
%       maxIter - Maximum number of iterations (default: 100).
%       tol      - Convergence tolerance (default: 1e-8).
%
%   OUTPUTS:
%       x         - Computed fixed point (column vector).
%       iter      - Number of iterations performed.
%       exitFlag  - Logical flag indicating if convergence was achieved.
%
%   EXAMPLE USAGE:
%       g = @(x) cos(x);
%       x0 = .5;
%       x = fixedPointIteration(g, x0);

% default values
if nargin < 3 || isempty(tol), tol = 1e-8; end
if nargin < 4 || isempty(maxIter), maxIter = 100; end

x = x0(:); % Ensure column vector if x0 is a row vector
exitFlag = false;

for iter = 1:maxIter
    x_new = fun(x); % Fixed-point iteration step
    % Check for convergence
    if norm(x_new - x, inf) < tol
        exitFlag = true;
        x = x_new;
        return;
    end
    x = x_new;
end
end


