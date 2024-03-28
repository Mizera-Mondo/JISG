N = 40;
Ns = 15; % Number of accessible nodes in each sample vector
E = N*2;
Ls = 1000; % Length of signal

A = rand_ugraph(N, E, 0.3, 0.2);
stmls = randn(N, Ls);
Y = (eye(N) - A)\stmls;
M = cell(1, Ls);
Z = cell(1, Ls);
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

[Aest, Yest] = JISG_SEM(Z, M, N, Ls, 0.3, 0.3, 2);
err = [norm(Aest - A)/norm(A); norm(Yest - Y)/norm(Y)];

%for la_1 = 0.33:0.03:0.5
%   for la_2 = 0.33:0.03:0.5
%       [Aest, Yest] = JISG_SEM(Z, M, N, Ls, la_1, la_2, 2);
%       err = [err, [norm(Aest - A)/norm(A); norm(Yest - Y)/norm(Y)]];
%   end
%end
%[Aest, Yest] = JISG_SEM_ADMM_only(Y, M, 20, 1000);
close all;
imagesc(A); title('Original'); figure; imagesc(Aest); title('Estimated');