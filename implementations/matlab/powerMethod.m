function [eigval, eigvec, iter, exitFlag] = powerMethod(A, z0, maxIter, tol, normType)
%POWERMETHOD    Power method (Von-Mises-Iteration) for dominant eigenpair.
%   [EIGVAL, EIGVEC, ITER, EXITFLAG] = POWERMETHOD(A, Z0, MAXITER, TOL, NORMTYPE)
%   computes the dominant eigenvalue and eigenvector of matrix A.
%
%   INPUTS:
%       A        - Square matrix (n x n).
%       z0       - Initial vector (n x 1), should be nonzero (default: randn(n,1)).
%       maxIter  - Maximum number of iterations (default: 100).
%       tol      - Convergence tolerance for eigenvalue (default: 1e-8).
%       normType - Norm for normalization: 2 or 'inf' (default: 2).
%
%   OUTPUTS:
%       eigval   - Dominant eigenvalue estimate.
%       eigvec   - Corresponding eigenvector (normalized).
%       iter     - Number of iterations performed.
%       exitFlag - Logical flag indicating if convergence was achieved.
%
%   EXAMPLE USAGE:
%       A = [2 1; 1 3];
%       [eigval, eigvec] = powerMethod(A);

% Default values
if nargin < 2 || isempty(z0), z0 = randn(size(A,1),1); end
if nargin < 3 || isempty(maxIter), maxIter = 100; end
if nargin < 4 || isempty(tol), tol = 1e-8; end
if nargin < 5 || isempty(normType), normType = 2; end

z = z0(:); % Ensure column vector if z0 is a row vector
z = z / norm(z,normType);

eigval = 0;
exitFlag = false;

for iter = 1:maxIter
    z_tilde = A * z;
    % Choose index k as the index of the largest absolute component
    [~, k] = max(abs(z_tilde));
    % Estimate eigenvalue using component k
    eigval_new = z_tilde(k) / z(k);
    % Normalize z_tilde
    z_new = z_tilde / norm(z_tilde,normType);
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