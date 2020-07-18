function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% ONEVSALL trains multiple logistic regression classifiers and returns
% all the classfiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i

% some useful variables
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

%  Add ones to the X data matrix
X = [ones(m, 1), X];

% set initial_theta
initial_theta = zeros(n + 1, 1);

% set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50000000000000);

% one-versus-many implementation using for loop

for  c=1:num_labels,
all_theta(c, :) = (fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    initial_theta, options))';
end

