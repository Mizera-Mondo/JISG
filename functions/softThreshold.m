function X = softThreshold(X, tau)
%SOFTTHRESHOLD 此处显示有关此函数的摘要
%   此处显示详细说明
X(abs(X) <= tau) = 0;
X(X > tau) = X(X > tau) - tau;
X(X < -tau) = X(X < -tau) + tau;
end
