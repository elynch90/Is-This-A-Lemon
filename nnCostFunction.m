function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%========== Forward Prop =========
%========== INPUT LAYER ==========
% activate units at input layer
a1 = X;
% add bias units at input layer
a1 = [ones(m, 1), a1];

%========== HIDDEN LAYER ==========
% activate units at hidden layer
% activation of units in hidden layer
z2 = a1 * Theta1';
% prediction at hidden layer
a2 = sigmoid(z2);
% add bias units at hidden layer
a2 = [ones(m, 1), a2];

%========== OUTPUT LAYER ==========
% activate output layer
z3 = a2 * Theta2';
% label prediciton for output layer
a3 = sigmoid(z3);

%========== Calculate Cost ==========
hx = a3;
%========== format output label for neural network cost function ==========
%  broadcast / scalar to binary vector representation
%yNew = zeros(m, num_labels);
yNew = round(a3);
%for i=1:size(y, 1),
%    yNew(i, y(i)) = 1;
%end

% octave's one liner efficient implementation of the broadcast function
%yNew = bsxfun(@eq, y, 1);

% calculate cost
J = (1 / m) * sum(sum(-yNew .* log(hx) - (1 - yNew) .* log(1 - hx)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

%========== Backpropagation ==========
%========== backprop at output layer ==========
delta3 = a3 - yNew;

%========== backprop at hidden layer ==========


% g'(z) or the sigmoid gradient function can be computed with g(z) * (1 - g(z))
% unvectorized: delta2 = (Theta2 * delta3 .* g(z) * (1 - g(z))

% second attempt at vectorization
%delta2 = ((Theta2 .^2)' * delta3') * (sigmoid(z2()) * (1 - sigmoid(z2))');

% vectorized as per reading material
%delta2 = (Theta2' * delta3') * (z2 * (1 - z2)'); % failed
%delta2 = (Theta2' * delta3') * sigmoidGradient(z2); % failed
delta2 = (Theta2' * delta3')' * [ones(size(z2, 1), 1), sigmoidGradient(z2)]'; % failed

% derive the sigmoid gradient
%sigmoidGrad = sigmoid(z2) * (1 - sigmoid(z2))'; % failed
%delta2 = (Theta2' * delta3') * sigmoidGrad; % failed

% attempt as per the lecture notes
%delta2 = (Theta2' * delta3') * (a2 * (1 - a2)'); % failed

% correct implementation with bias units and dot matrix multiplication
delta2 = (delta3 * Theta2) .* [ones(size(z2, 1),1) sigmoidGradient(z2)];

% discard s2 zero
delta2 = delta2(:, 2:end);

% accumulate gradients
delta2Sum = delta3' * a2;
delta1Sum = delta2' * a1;

% unregularized
Theta1_grad = (1 / m) * delta1Sum;
Theta2_grad = (1 / m) * delta2Sum;
                                 
%========== return theta gradients ==========
% bias units remain unregularized
%Theta1_grad(:, 1) = (1 / m) * delta1Sum(1, :);
%Theta2_grad(:, 1) = (1 / m) * delta2Sum(1, :);
                                
% calculate regularization coefficients, the zeros in column 1 prevent the bias unit from being altered
Theta1_gradReg = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_gradReg = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad += Theta1_gradReg;
Theta2_grad += Theta2_gradReg;
                                 
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% calculute regularization term and remembering to splice the matrices as to avoid the bias terms
regTerm = (lambda/(2*m)) * [sum(sum(Theta1(1:end, 2:end) .^2)) + sum(sum(Theta2(1:end, 2:end) .^2))];

% add regularization term to NN cost function J
J += regTerm;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

