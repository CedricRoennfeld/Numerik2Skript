function [eigval, eigvec, iter, exitFlag] = inverseIteration(A, lambda0, z0, maxIter, tol, normType)
%INVERSEITERATION    Inverse Iteration (Wielandt) for eigenpair approximation.
%   [EIGVAL, EIGVEC, ITER, EXITFLAG] = INVERSEITERATION(A, LAMBDA0, Z0, MAXITER, TOL, NORMTYPE)
%   computes an eigenvalue and eigenvector of matrix A closest to the shift LAMBDA0.
%
%   INPUTS:
%       A        - Square matrix (n x n).
%       lambda0  - Shift (a priori eigenvalue estimate).
%       z0       - Initial vector (n x 1), should be nonzero (default: randn(n,1)).
%       maxIter  - Maximum number of iterations (default: 100).
%       tol      - Convergence tolerance for eigenvalue (default: 1e-8).
%       normType - Norm for normalization: 2 or 'inf' (default: 2).
%
%   OUTPUTS:
%       eigval   - Approximated eigenvalue.
%       eigvec   - Corresponding eigenvector (normalized).
%       iter     - Number of iterations performed.
%       exitFlag - Logical flag indicating if convergence was achieved.
%
%   EXAMPLE USAGE:
%       A = [2 1; 1 3];
%       [eigval, eigvec] = inverseIteration(A, 2.5);

% Default values
if nargin < 3 || isempty(z0), z0 = randn(size(A,1),1); end
if nargin < 4 || isempty(maxIter), maxIter = 100; end
if nargin < 5 || isempty(tol), tol = 1e-8; end
if nargin < 6 || isempty(normType), normType = 2; end

n = size(A,1);
z = z0(:);
z = z / norm(z, normType);

eigval = lambda0;
exitFlag = false;

I = eye(n);

for iter = 1:maxIter
    % Solve (A - lambda0*I) * z_tilde = z
    z_tilde = (A - lambda0*I) \ z;
    % Normalize
    z_new = z_tilde / norm(z_tilde, normType);
    % Choose index k as the largest absolute component
    [~, k] = max(abs(z_new));
    % Compute eigenvalue approximation
    denom = (A - lambda0*I) * z_new;
    mu_t = z_new(k) / denom(k);
    eigval_new = lambda0 + 1 / mu_t;
    % Check for convergence
    if iter > 1 && abs(eigval_new - eigval) < tol
        exitFlag = true;
        break;
    end
    eigval = eigval_new;
    z = z_new;
end

eigvec = z;

end