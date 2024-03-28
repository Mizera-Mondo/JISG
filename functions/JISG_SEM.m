function [A, Y] = JISG_SEM(Z,M,n,l, la_1, la_2, rho)
%% JISG: Semi-Blind Inference of Topologies and Dynamical Processes Over Dynamic Graphs
% SEM: Based on SEM settings

% Z: signal samples. Z is not regular sized, meaning z may have different length over its
% columns. Since what it's assigned to be a vector cell with length of l, not a matrix.
% M: Also a l-by-one cell. Matrixs that maps original signal yt into
% semi-blind samples zt.
% n: Size of underlying graph.
% l: Length of signal.
% la_1: 1st regularization parametre of ADMM process. Controlling,
% generally, the sparsity of result.
% la_2: 2st regularization parametre of ADMM process. Forcing the result to
% have smaller norm.
% rho: Augument Lagrangian parametre. 

% A: Estimated topology of underlying graph, in the form of adjacency.
% Y: Recovered original signal matrix.

%% Initialization
Y = zeros(n,l);
for i = 1:l
    Y(:, i) = (M{i})'*Z{i};
end
E = eye(n);
A = zeros(n);
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
while ~convg && ~maxiterReached
    iteridx = iteridx + 1;
    A_LAST = A;
    Y_LAST = Y;
    % Update of A, Using ADMM
    R = Y*Y';
%    la_1 = 5;
%    la_2 = 5;
 %   rho = 2;
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
        rho = 1.05*rho; % Update of rho
        
        %===== ADMM convg condition here =====
        deltaNormA = norm(A - Alas)/norm(Alas);
        ADMMconvg = deltaNormA < tol;
        ADMMiter = ADMMiter + 1;
        ADMMiterReached = ADMMiter > maxiter;
    end
    
    % Update of y
    
    for t = 1:l
        yconvg = false;
        
        while ~yconvg
            [mt, ~] = size(M{t});
            Mt = M{t};
            zt = Z{t};
            R = E - A;
            yt = Y(:,t);
            cost_y = @(y) mt/mu*norm((E - A)*y)^2 + norm(zt - Mt*y)^2;
            c_value = cost_y(yt);
            gd = (mt/mu*(R'*R) + Mt'*Mt)*yt - Mt'*zt; % Gradient Direction
            
            %=====Stepsize selection here=====
            % Using Armijo rule
            dr = norm(gd)^2; % Descending rate, linear
            beta = 0.5;
            bp = 0;
            acc_theta = false;
            
            while bp < 10 && ~acc_theta
                theta = beta^bp;
                step = theta*gd;
                ad = c_value - cost_y(yt - step); % Actual descend
                pd = theta*dr; % Predicted descend
                acc_theta = ad > alpha*pd;
                bp = bp + 1;
            end
            
            %===== y convg condition here =====
            
            if bp == 10 && ~acc_theta % Which indicates yt is probably at a stationary point
                yconvg = true;
            else
                yconvg = (norm(step)/norm(Y(:,t))) < tol; % Utilizing yt instead of Y(:,t) is problematic here, as Y is initialized as zero matrix and will cuz DIV-BY-ZERO
            end
            
            if ~yconvg
                Y(:,t) = yt - step;
            end
        end
    end
    %===== General Convg Condition =====
    deltaY = norm(Y - Y_LAST)/norm(Y_LAST);
    deltaA = norm(A - A_LAST)/norm(A_LAST);
    disp(['iter  ' num2str(iteridx)]);
    disp(['Delta A ' num2str(deltaA)]);
    disp(['Delta Y ' num2str(deltaY)]);
    convg = deltaY < tol && deltaA < tol;
    maxiterReached = iteridx >= maxiter;
end
    
%%
end
