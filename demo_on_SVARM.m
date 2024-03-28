N = 40;
Ns = 15; % Number of accessible nodes in each sample vector
E = 40;
Ls = 10000; % Length of signal

A1 = rand_ugraph(N, E, 0.3, 0.2);
A2 = rand_digraph(N, E, 0.1, 0.2);
A2 = A2 + diag(0.05*rand(N,1));
stmls = randn(N, Ls);
Y = zeros(N, Ls);
Y(:, 1) = stmls(:, 1);
for i = 2:Ls
    Y(:, i) = (eye(N) - A1)\(A2*Y(:, i - 1) + stmls(:, i));
end
M = cell(Ls, 1);
Z = cell(Ls, 1);
%M1 = [eye(15), zeros(15, 5)];
%M2 = [zeros(15, 5), eye(15)];
%for i = 1:500
%    Z{2*i - 1} = M1*Y(:, 2*i - 1);
%    Z{2*i} = M2*Y(:, 2*i);
%    M{2*i - 1} = M1;
%    M{2*i} = M2;
%end
for i = 1:Ls
    Mi = zeros(Ns, N);
    k = randperm(N, Ns);
    for j = 1:Ns
        Mi(j, k(j)) = 1;
    end
    M{i} = Mi;
    Z{i} = Mi*Y(:, i);
end

[A1est, A2est, Yest] = JISG_SVARM(Z, M, [0.5, 0.5; 5, 0.1], 2);
err = [norm(A1est - A1)/norm(A1); norm(A2est - A2est/norm(A2)); norm(Yest - Y)/norm(Y)];

%for la_1 = 0.33:0.03:0.5
%   for la_2 = 0.33:0.03:0.5
%       [Aest, Yest] = JISG_SEM(Z, M, N, Ls, la_1, la_2, 2);
%       err = [err, [norm(Aest - A)/norm(A); norm(Yest - Y)/norm(Y)]];
%   end
%end
%[Aest, Yest] = JISG_SEM_ADMM_only(Y, M, 20, 1000);
close all;
imagesc(A1); title('Original A1'); 
figure; imagesc(A1est); title('Estimated A1');
figure; imagesc(A2); title('Original A2'); 
figure; imagesc(A2est); title('Estimated A2');