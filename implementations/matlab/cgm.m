function [x, iter, exitFlag] = cgm(A, b, x0, maxIter, tol)
%CGM    Conjugate Gradient Method for solving Ax = b with Hermitian, positive definite A.
%   [X, ITER, EXITFLAG] = CGM(A, B, X0, MAXITER, TOL) solves the linear system
%   A*X = B for X using the Conjugate Gradient Method, where A is Hermitian and
%   positive definite (complex numbers allowed).
%
%   INPUTS:
%       A        - Hermitian, positive definite matrix (n x n, complex allowed)
%       b        - Right-hand side vector (n x 1, complex allowed)
%       x0       - Initial guess (n x 1, complex allowed)
%       maxIter  - Maximum number of iterations (default: 100)
%       tol      - Convergence tolerance for norm(r) (default: 1e-8)
%
%   OUTPUTS:
%       x        - Computed solution vector (n x 1)
%       iter     - Number of iterations performed
%       exitFlag - Logical flag indicating if convergence was achieved
%
%   EXAMPLE USAGE:
%       A = [17, 2; 2, 7];
%       b = [2; 2];
%       x0 = [0; 0];
%       x = cgm(A, b, x0);

% default values
if nargin < 3 || isempty(x0), x0 = zeros(size(A,1),1); end
if nargin < 4 || isempty(maxIter), maxIter = 100; end
if nargin < 5 || isempty(tol), tol = 1e-8; end

x = x0(:); % Ensure column vector if x0 is a row vector
% Initialize variables
r = b - A*x;
d = r;
exitFlag = false;

for iter = 1:maxIter
    Ad = A*d;
    alpha = (r'*r) / (d'*Ad);   % Step size
    x = x + alpha*d;            % Update solution
    r_new = r - alpha*Ad;       % Update residual
    % Check for convergence
    if norm(r_new, 2) < tol
        exitFlag = true;
        return;
    end
    beta = (r_new'*r_new) / (r'*r);
    d = r_new + beta*d;         % Update direction
    r = r_new;
end
end