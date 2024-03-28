function [A1, A2, Y] = JISG_SVARM(Z, M, La, rho)
%% JISG_SVARM JISG algorithm for graph topology inference within SVARM
% setting. Estimating both A1 and A2 adjacency with A2 allowing diagonal
% elements; also recover original signal Y under smoothness metric.
% Estimating A1 & A2 with ADMM, then Y with GD.

% Input parametres
% Z: l-by-1 cells. Corrupted signal samples vectors, each shorter than orignal signal vectors.
% M: l-by-1 cells. Zi = Mi*Yi.
% La: 2-by-2 matrix, consisting of egularization parametres. 
%   La_11: 1st regularization parametre of A1, leading to more sparse
%   solution.
%   La_12: 2st regularization parametre of A1, forcing solution to have
%   smaller F-norm.
%   La_21: Same as la_11, of A2.
%   La_22: Same as la_12, of A2.
% rho: Starting up value of augmented Lagrangian parametre.

% Output variables
% A1: Instant adjacency matrix, estimated.
% A2: Lagged adjacency, allowing diagonal elements.
% Y: Recovered signal matrix.

%% Initialization
[l, ~] = size(Z); % l: Length of Signal
[~, n] = size(M{1}); % n: Node count
I = eye(n);
A1 = I;
Psi1 = I; % Auxilary variable for ADMM
A2 = I;
Psi2 = I;
Y = zeros(n, l);
tol = 1e-3;
for i = 1:l
    Y(:, i) = (M{i})'*Z{i}; % Starting point of signal Matrix
end

%% Estimating Iterations
isConvg = false;
isMaxIterReached = false;
iterCount = 0;
maxIter = 1000;
while ~isConvg && ~isMaxIterReached
    Y_LAST = Y;
    A1_LAST = A1;
    A2_LAST = A2;
    % ADMM Iterations, solving A1 & A2 simultaneously
    U1 = eye(n);
    U2 = U1; % Lagrangian multipliers
    Yt = Y(:, 2:l);
    Yt_1 = Y(:, 1:l - 1);
    Rt = Yt*Yt';
    C = Yt*Yt_1';
    Rt_1 = Yt_1*Yt_1';

    isAdmmConvg = false;
    isAdmmMaxIterReached = false;
    admmIterCount = 0;

    while ~isAdmmConvg && ~isAdmmMaxIterReached

        prevA1 = A1;
        prevA2 = A2;

        % Updating A1, A2
        Ka1 = Rt + (La(1,2) + rho)*I;
        Fa1 = Rt - A2*C' - U1 + rho*Psi1;
        A1 = Fa1/Ka1;
        Ka2 = Rt_1 + (La(2,2) + rho)*I;
        Fa2 = C - A1*C - U2 + rho*Psi2; % Beware of A1*C, it somehow doesn't take C's transposition.
        A2 = Fa2/Ka2;

        % Updating Psi1, Psi2
        tau1 = La(1,1)/rho; % Soft threshold parametre
        Au1 = softThreshold(A1 + 1/rho*U1, tau1);
        Psi1 = Au1 - diag(diag(Au1));
        tau2 = La(2,1)/rho;
        Psi2 = softThreshold(A2 + 1/rho*U2, tau2);

        % Updating U1, U2
        U1 = U1 + rho*(A1 - Psi1);
        U2 = U2 + rho*(A2 - Psi2);
        rho = 1.03*rho;
        
        % Convg condition check
        deltaA1 = norm(A1 - prevA1)/norm(prevA1);
        deltaA2 = norm(A2 - prevA2)/norm(prevA2);
        admmIterCount = admmIterCount + 1;
        isAdmmConvg = deltaA1 < tol && deltaA2 < tol;
        isAdmmMaxIterReached = admmIterCount > maxIter;
    end

    %% RTS Smoother for recovering, recovering signal Y vector-wisely
    % The algorithm does not require any iterations
    
    F = (I - A1)\A2;
    Q = inv((I - A1)'*(I - A1));
    mu = 1;

    sigForward = cell(l,1);
    sigCurrent = cell(l,1);
    sigCurrent{1} = I;

    yForward = zeros(n, l);
    yCurrent = zeros(n, l);
    yCurrent(:, 1) = Y(:, 1);
    % Kalman filter part
    for i = 2:l
        yForward(:, i) = F*yCurrent(:, i - 1);
        sigForward{i} = F*sigCurrent{i-1}*F' + Q;
        Mt = M{i};
        [mt, ~] = size(Mt);
        G = sigForward{i}*(Mt)'/((mt/mu)*eye(mt) + Mt*sigForward{i}*Mt');
        sigCurrent{i} = (I - G*Mt)*sigForward{i};
        yCurrent(:, i) = yForward(:, i) + G*(Z{i} - Mt*yForward(:, i));
    end

    % kalman smoother part
    for i = l - 1:-1:1
        G = sigCurrent{i}*F'/sigForward{i + 1};
        Y(:, i) = yCurrent(:, i) + G*(Y(:, i + 1) - yForward(:, i + 1));
    end

    %% Convg condition check
    iterCount = iterCount + 1;
    deltaY = norm(Y - Y_LAST)/norm(Y_LAST);
    deltaA1 = norm(A1 - A1_LAST)/norm(A1_LAST);
    deltaA2 = norm(A2 - A2_LAST)/norm(A2_LAST);
    disp(['iter  ' num2str(iterCount)]);
    disp(['Delta A1 ' num2str(deltaA1)]);
    disp(['Delta A2 ' num2str(deltaA2)]);
    disp(['Delta Y ' num2str(deltaY)]);
    isConvg = deltaY < tol && deltaA1 < tol && deltaA2 < tol;
    isMaxIterReached = iterCount >= maxIter;
end
end