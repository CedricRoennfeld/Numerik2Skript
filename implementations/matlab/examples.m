%%      Fixed-Point Iteration Example
clear;
g = @(x) cos(x);
x0 = 0.5;
[x, iter] = fixedPointIteration(g, x0);
fprintf('Fixed-Point Iteration Example:\n');
fprintf('Function: g(x) = cos(x)\n');
fprintf(['Initial guess: [', repmat(' %.4f', 1, size(x0,1)), ' ]\n'], x0);
fprintf(['Solution: [', repmat(' %.4f', 1, size(x,1)), ' ]\n'], x);
fprintf('Residual norm |g(x) - x|: %.4e\n', norm(g(x) - x));
fprintf('Iterations: %d\n\n\n', iter);



%%      Newton's Method Example
clear;
fun = @(x) [x(1)^2 + x(2)^2 - 1; x(1) - x(2)];
jac = @(x) [2*x(1), 2*x(2); 1, -1];
x0 = [0.5; 0.5];
[x, ~, iter] = newtonMethod(fun, jac, x0);
fprintf('Newton''s Method Example:\n');
fprintf('Function: f(x) = [x(1)^2 + x(2)^2 - 1; x(1) - x(2)]\n');
fprintf('Jacobian: J(x) = [2*x(1), 2*x(2); 1, -1]\n');
fprintf(['Initial guess: [', repmat(' %.4f', 1, size(x0,1)), ' ]\n'], x0);
fprintf(['Solution: [', repmat(' %.4f', 1, size(x,1)), ' ]\n'], x);
fprintf('Residual norm |f(x)|: %.4e\n', norm(fun(x)));
fprintf('Iterations: %d\n\n\n', iter);



%%      Jacobi Method Example
clear;
A = [2, 0, 1; 1, -4, 1; 0, -1, 2];
B = [1; 4; -1];
x0 = [1; 1; 1];
[X, iter] = jacobiMethod(A, B, x0);
fprintf('Jacobi Method Example:\n');
fprintf('Coefficient matrix A:\n');
for i = 1:size(A, 1)
    fprintf('  [%.4f', A(i, 1));
    for j = 2:size(A, 2)
        fprintf(' %.4f', A(i, j));
    end
    fprintf(' ]\n');
end
fprintf('Right-hand side B:\n');
fprintf(['B = [', repmat(' %.4f', 1, size(B,1)), ']\n'], B');
fprintf(['Initial guess: [', repmat(' %.4f', 1, size(x0,1)), ' ]\n'], x0);
fprintf(['Solution: [', repmat(' %.4f', 1, size(X,1)), ' ]\n'], X);
fprintf('Residual norm |A*X - B|: %.4e\n', norm(A*X - B));
fprintf('Iterations: %d\n\n\n', iter);



%%      Gauss-Seidel Method Example
clear;
A = [2, 0, 1; 1, -4, 1; 0, -1, 2];
B = [1; 4; -1];
x0 = [1; 1; 1];
[X, iter] = gaussSeidelMethod(A, B, x0);
fprintf('Gauss-Seidel Method Example:\n');
fprintf('Coefficient matrix A:\n');
for i = 1:size(A, 1)
    fprintf('  [%.4f', A(i, 1));
    for j = 2:size(A, 2)
        fprintf(' %.4f', A(i, j));
    end
    fprintf(' ]\n');
end
fprintf('Right-hand side B:\n');
fprintf(['B = [', repmat(' %.4f', 1, size(B,1)), ']\n'], B');
fprintf(['Initial guess: [', repmat(' %.4f', 1, size(x0,1)), ' ]\n'], x0);
fprintf(['Solution: [', repmat(' %.4f', 1, size(X,1)), ' ]\n'], X);
fprintf('Residual norm |A*X - B|: %.4e\n', norm(A*X - B));
fprintf('Iterations: %d\n\n\n', iter);



%%      Conjugate Gradient Method Example
clear;
A = [17, 2; 2, 7];
b = [2; 2];
x0 = [0; 0];
[x, iter] = cgm(A, b, x0);
fprintf('Conjugate Gradient Method Example:\n');
fprintf('Coefficient matrix A:\n');
for i = 1:size(A, 1)
    fprintf('  [%.4f', A(i, 1));
    for j = 2:size(A, 2)
        fprintf(' %.4f', A(i, j));
    end
    fprintf(' ]\n');
end
fprintf('Right-hand side B:\n');
fprintf(['B = [', repmat(' %.4f', 1, size(b,1)), ']\n'], b');
fprintf(['Initial guess: [', repmat(' %.4f', 1, size(x0,1)), ' ]\n'], x0);
fprintf(['Solution: [', repmat(' %.4f', 1, size(x,1)), ' ]\n'], x);
fprintf('Residual norm |A*X - B|: %.4e\n', norm(A*x - b));
fprintf('Iterations: %d\n\n\n', iter);



%%      Preconditioned Conjugate Gradient Method Example
clear;
A = [1e+8, 0; 0, 1];
M = diag(diag(A));
b = [1; 1];
x0 = [0; 0];
tol = 1e-12;
[x, iter] = pcgm(A, M, b, x0, 100, tol);
fprintf('Preconditioned Conjugate Gradient Method Example:\n');
fprintf('Coefficient matrix A:\n');
for i = 1:size(A, 1)
    fprintf('  [%.1e', A(i, 1));
    for j = 2:size(A, 2)
        fprintf(' %.1e', A(i, j));
    end
    fprintf(' ]\n');
end
fprintf('Preconditioner M:\n');
for i = 1:size(M, 1)
    fprintf('  [%.1e', M(i, 1));
    for j = 2:size(M, 2)
        fprintf(' %.1e', M(i, j));
    end
    fprintf(' ]\n');
end
fprintf('Right-hand side B:\n');
fprintf(['B = [', repmat(' %.4f', 1, size(b,1)), ']\n'], b');
fprintf(['Initial guess: [', repmat(' %.4f', 1, size(x0,1)), ' ]\n'], x0);
fprintf(['Solution: [', repmat(' %.4e', 1, size(x,1)), ' ]\n'], x);
fprintf('Residual norm |A*X - B|: %.4e\n', norm(A*x - b));
fprintf('Iterations: %d\n\n\n', iter);



%%     Power Method Example
clear;
A = [2, 1; 1, 3];
z0 = [1; 1];
[eigval, eigvec, iter] = powerMethod(A, z0);
fprintf('Power Method Example:\n');
fprintf('Matrix A:\n');
for i = 1:size(A, 1)
    fprintf('  [%.4f', A(i, 1));
    for j = 2:size(A, 2)
        fprintf(' %.4f', A(i, j));
    end
    fprintf(' ]\n');
end
fprintf(['Initial vector: [', repmat(' %.4f', 1, size(z0,1)), ' ]\n'], z0);
fprintf(['Eigenvalue: %.4f\n', repmat(' %.4f', 1, size(eigval,1)), ' ]\n'], eigval);
fprintf(['Eigenvector: [', repmat(' %.4f', 1, size(eigvec,1)), ' ]\n'], eigvec);
fprintf('Residual norm |Ax - lambda*x|: %.4e\n', norm(A*eigvec - eigval*eigvec));
fprintf('Iterations: %d\n\n\n', iter);



%%     Inverse Iteration Example
clear;
A = [2, 1; 1, 3];
lambda0 = 3.5;  
z0 = [1; 1];
[eigval, eigvec, iter] = inverseIteration(A, lambda0, z0);
fprintf('Inverse Iteration Example:\n');
fprintf('Matrix A:\n');
for i = 1:size(A, 1)
    fprintf('  [%.4f', A(i, 1));
    for j = 2:size(A, 2)
        fprintf(' %.4f', A(i, j));
    end
    fprintf(' ]\n');
end
fprintf('Initial guess: %d\n', lambda0);
fprintf(['Initial vector: [', repmat(' %.4f', 1, size(z0,1)), ' ]\n'], z0);
fprintf(['Eigenvalue: %.4f\n', repmat(' %.4f', 1, size(eigval,1)), ' ]\n'], eigval);
fprintf(['Eigenvector: [', repmat(' %.4f', 1, size(eigvec,1)), ' ]\n'], eigvec);
fprintf('Residual norm |Ax - lambda*x|: %.4e\n', norm(A*eigvec - eigval*eigvec));
fprintf('Iterations: %d\n\n\n', iter);