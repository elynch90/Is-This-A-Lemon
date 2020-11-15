% here we will use a single class classifier to determine whether
% a given image is of a lemon

% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 2; % 2 labels, from 1 to 2



% ========== Loading and Visualizing the Data ==========
% load the training data
fprintf('Loading and Visualizing Data ...\n')

% process the images or use this default method if you can process the pictures beforehand
load('lemonData.mat'); % training data stored in arrays X, y
m = size(X, 1);

% randomly select 10 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:10), :)

displayData(sel);
fprintf('Program Paused. Press enter to continue.\n');
pause;

% =========Split the data into Training, Testing and Validation sets ==========
% the data is split us ing the 60-20-20 distribution ratio
X = X(1: ceil(M * 0.6));
y = y(1: ceil(M *0.6));
Xval = X(ceil(M * 0.6)  + 1: ceil(M * 0.8));
yval = y(ceil(M * 0.6) + 1: ceil(M * 0.8));
Xtest = X(ceil(M * 0.8) + 1:end);
ytest = y(ceil(M * 0.8) + 1:end);

%% ================ Part 2: Loading Parameters ================
% initialize Theta with symetry breaking
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% unroll the parameters
nn_parameters = [Theta1(:); Theta2(:)]
fprintf('\nFeed Forward Using Neural Network...\n')

% Weight regularization parameter (we set this to 0 here)
lambda = 0;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 5000);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by
%  displaying the hidden units to see what features they are capturing in
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%[all_theta] = oneVsAll(X, y, num_labels, lambda); % train the model
%fprintf('Program paused. Press enter to continue.\n');
%pause;
               
%========= Validation =========
%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function.
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- Figure 3 in ex5.pdf
%

lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%

//lambda = 0;
//[theta] = trainLinearReg(X_poly, y, lambda);
//
//% Plot training data and fit
//figure(1);
//plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
//plotFit(min(X), max(X), mu, sigma, theta, p);
//xlabel('Change in water level (x)');
//ylabel('Water flowing out of the dam (y)');
//title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
//
//figure(2);
//[error_train, error_val] = ...
//    learningCurve(X_poly, y, X_poly_val, yval, lambda);
//plot(1:m, error_train, 1:m, error_val);
//
//title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
//xlabel('Number of training examples')
//ylabel('Error')
//axis([0 13 0 100])
//legend('Train', 'Cross Validation')
//
//fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
//fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
//for i = 1:m
//    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
//end
//
//fprintf('Program paused. Press enter to continue.\n');
//pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
    fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;


% test the model
% add zeros to validation set
%valTestSet = [ones(size(valTestSet, 1), 1) valTestSet]
%pred = sigmoid(valTestSet * [Theta1; Theta2]');
%pred = predict(Theta1, Theta2, valTestSet);

fprintf('\nValidation Set Accuracy: %f\n', mean(double(pred == valLabels)) * 100);
pause;


