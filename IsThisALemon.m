% here we will use a single class classifier to determine whether
% a given image is of a lemon

% set up the parameters which will be used in training the classifier
input_layer_size = 400; % 20x20 Input Images of Digits
num_labels = 1; % since this is only a single class classifier

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

% ========== Vectorized Logisitc Regression =========
% ========== One-vs-All Training=========
fprintf('\nOne-vs-All Logistic Regression...\n');

lambda = 0.01;
[all_theta] = oneVsAll(X, y, num_labels, lambda); % train the model
fprintf('Program paused. Press enter to continue.\n');
pause;

%========== Prediction for Binary Classification =========
% add zeros
X_t = [zeros(m, 1) X]  
pred = sigmoid(X_t * all_theta');
p = round(pred);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;
               
%========= Validation =========
% test the model
% add zeros to validation set
valTestSet = [ones(size(valTestSet, 1), 1) valTestSet]
pred = sigmoid(valTestSet * all_theta');
pVal = round(pred);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pVal(1:10, 1) == valLabels)) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;

%========== Predict for One-Vs-All =========

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

