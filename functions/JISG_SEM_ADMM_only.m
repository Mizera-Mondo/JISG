function [A, Y] = JISG_SEM_ADMM_only(Y,M,n,l)
%% JISG: Semi-Blind Inference of Topologies and Dynamical Processes Over Dynamic Graphs
% SEM: Based on SEM settings

% Z: signal samples. Z is not regular sized, meaning z may have different length over its
% columns. Since what it's assigned to be a vector cell with length of l, not a matrix.
% M: Also a l-by-one cell. Matrixs that maps original signal yt into
% semi-blind samples zt.
% n: Size of underlying graph.
% l: Length of signal.

% A: Estimated topology of underlying graph, in the form of adjacency.
% Y: Recovered original signal matrix.

%% Initialization

E = eye(n);
A = E;
Psi = A;
U = A;
mu = 0.5;
tol = 1e-3;

alpha = 1e-4;

convg = false;
maxiter = 1e3;
maxiterReached = false;
iteridx = 0;

%% Estimation Iterations
    % Update of A, Using ADMM
    R = Y*Y';
    la_1 = 0.5;
    la_2 = 1;
    rho = 2;
    ADMMconvg = false;
    ADMMiter = 0;
    ADMMiterReached = false;
    while ~ADMMconvg && ~ADMMiterReached
        
        Alas = A; % Storing last value of A
        % ADMM Iteration, updating A
        Ka = R + (la_2 + rho)*E;
        Ya = R - U + rho*(Psi - diag(diag(Psi)));
        A = Ya/Ka; % The closed-form expression of A
        
        Au = A + 1/rho*U;
        Au(abs(Au) < la_1/rho) = 0;
        Au(Au > la_1/rho) = Au(Au > la_1/rho) - la_1/rho;
        Au(Au < -la_1/rho) = Au(Au < -la_1/rho) + la_1/rho;
        
        Psi = Au - diag(diag(Au)); % The closed-form expression of auxilary variable Psi
        U = U + rho*(A - Psi); % Update of Lagrange multiplier 
        rho = 1.01*rho; % Update of rho
        
        %===== ADMM convg condition here =====
        deltaNormA = norm(A - Alas)/norm(Alas);
        ADMMconvg = deltaNormA < tol;
        ADMMiter = ADMMiter + 1;
        ADMMiterReached = ADMMiter > maxiter;
    end
    
%%
end
