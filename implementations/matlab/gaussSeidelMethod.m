function [X, iter, exitFlag] = gaussSeidelMethod(A, B, x0, maxIter, tol)
%GAUSSSEIDELMETHOD    Gauss-Seidel iterative method for solving AX = B.
%   [X, ITER, EXITFLAG] = GAUSSSEIDELMETHOD(A, B, X0, MAXITER, TOL)
%   solves the linear system(s) AX = B using the Gauss-Seidel method.
%
%   INPUTS:
%       A        - Coefficient matrix (n x n).
%       B        - Right-hand side (n x m), each column is a system.
%       x0       - Initial guess (n x m) (default: zeros).
%       maxIter  - Maximum number of iterations (default: 100).
%       tol      - Convergence tolerance (default: 1e-8).
%
%   OUTPUTS:
%       X        - Solution(s) (n x m).
%       iter     - Number of iterations performed.
%       exitFlag - Logical flag indicating if convergence was achieved.
%
%   EXAMPLE USAGE:
%       A = [2, 0, 1; 1, -4, 1; 0, -1, 2];
%       B = [1; 4; -1];
%       x0 = [1; 1; 1];
%       X = gaussSeidelMethod(A, B, x0);

% default values
if nargin < 3 || isempty(x0), x0 = zeros(size(A,1), size(B,2)); end
if nargin < 4 || isempty(maxIter), maxIter = 50; end
if nargin < 5 || isempty(tol), tol = 1e-6; end

n = size(A,1);
m = size(B,2);
X = x0;
exitFlag = false;

for iter = 1:maxIter
    X_old = X;
    % Solve system (D - L) * X = U * X + B using back substitution
    % where D is the diagonal of A, L is the lower triangular part
    % and U is the upper triangular part.
    for i = 1:n
        sum1 = zeros(1, m);
        sum2 = zeros(1, m);
        if i > 1
            sum1 = A(i,1:i-1) * X(1:i-1,:);
        end
        if i < n
            sum2 = A(i,i+1:n) * X_old(i+1:n,:);
        end
        X(i,:) = (B(i,:) - sum1 - sum2) / A(i,i);
    end
    % Check for convergence
    if norm(X - X_old, inf) < tol
        exitFlag = true;
        return;
    end
end
end