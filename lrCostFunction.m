function [J, grad] = lrCostFunction(theta, X,  y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
% regularization

% Initialize some useful values
m = length(y); % number of labels

% initialize cost and gradient
J = 0;
grad = zeros(size(theta));

% logistic hypothesis given weights
hx = sigmoid(X * theta)

% count the number of features
n = columns(X)

% regularized cost function with array indexing to account for non regularization of theta 0

J = (-y' * log(hx) - (1-y') * log(1 - hx)) / m + (lambda / (2*m)) * sum(theta(2:n, :).^2)

% regularized gradient update
grad(1) = (X(:, 1)' * (hx - y)) / m;
           grad(2:n) = (X(:, 2:n)' * (hx - y) / m + (lambda / m) * theta(2:n, :));

% =============================================================

grad = grad(:);

end
