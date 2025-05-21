function [x, iter, exitFlag] = pcgm(A, M, b, x0, maxIter, tol)
%PCGM    Preconditioned Conjugate Gradient Method for solving Ax = b.
%   [X, ITER, EXITFLAG] = PCGM(A, M, B, X0, MAXITER, TOL) solves the linear system
%   A*X = B for X using the Preconditioned Conjugate Gradient Method, where A is
%   Hermitian and positive definite, and M is a Hermitian, positive definite preconditioner.
%
%   INPUTS:
%       A        - Hermitian, positive definite matrix (n x n)
%       M        - Preconditioner matrix (n x n, Hermitian, positive definite)
%       b        - Right-hand side vector (n x 1)
%       x0       - Initial guess (n x 1)
%       maxIter  - Maximum number of iterations (default: 100)
%       tol      - Convergence tolerance for norm(r) (default: 1e-8)
%
%   OUTPUTS:
%       x        - Computed solution vector (n x 1)
%       iter     - Number of iterations performed
%       exitFlag - Logical flag indicating if convergence was achieved
%
%   EXAMPLE USAGE:
%       A = [1e+8, 0; 0, 1];
%       M = diag(diag(A));
%       b = [1; 1];
%       x0 = [0; 0];
%       x = pcgm(A, M, b, x0);

% default values
if nargin < 4 || isempty(x0), x0 = zeros(size(A,1),1); end
if nargin < 5 || isempty(maxIter), maxIter = 100; end
if nargin < 6 || isempty(tol), tol = 1e-8; end

x = x0(:); % Ensure column vector if x0 is a row vector
% Initialize variables
r = b - A*x;
s = M \ r;
d = s;
exitFlag = false;

for iter = 1:maxIter
    Ad = A*d;
    alpha = (r'*s) / (d'*Ad);       % Step size
    x = x + alpha*d;                % Update solution
    r_new = r - alpha*Ad;           % Update residual
    if norm(r_new, 2) < tol
        exitFlag = true;
        break;
    end
    s_new = M \ r_new;              % Preconditioned residual
    beta = (r_new'*s_new) / (r'*s);
    d = s_new + beta*d;             % Update direction
    r = r_new;
    s = s_new;
end
end