% ========== Process the raw pngs from the training set folder and write to a .mat file ==========
% run this script when it is necissary to process or reprocess the raw training set
% TODO: optimization

pkg load image % import forge imag pkg
%========== load image matrix and label vector ==========
trainingFolderPos = dir('rawTrainingImgs/postive');
trainingFolderNeg = dir('rawTrainingImgs/negative');
validationFolderNeg = dir('validationImgs/negative');
validationFolderPos = dir('validationImgs/positive');

mPos = length(trainingFolderPos) - 3; % number of images in the positive training set excluding meta data
mNeg = length(trainingFolderNeg) - 3; % number of images in the negative training set excluding metadata
mValidationNeg = length(validationFolderNeg);
mValidationPos = length(validationFolderPos);
X = zeros(mPos + mNeg, 400); % initialize output matrix 400 columns from 20 x 20 images
valTestSet = zeros(mValidationNeg + mValidationPos, 400);

%========= preprocess training set images =========
% prepare negative training examples
printf('Preparing negative training examples...\n');

printf('Running...\n');
for i=4:mNeg, % for each image excluding metadata
    display(trainingFolderNeg(i).name)
    curImg = imread(strcat('rawTrainingImgs/negative/', trainingFolderNeg(i).name)); % current img itteration
    % convert  to greyscale
    greyscaleImg = rgb2gray(curImg);
    processedImg = imresize(greyscaleImg, [20, 20]);  % resize image to 20 x 20
   
    % reshape current image as vector
    processedImg = reshape(processedImg, [1, 400]);
    % store in output matrix
    X(i - 3, :) = processedImg;
end

% prepare positive training examples
printf('Preparing positive training examples...\n');

printf('Running...\n');
for i=4:mPos, % for each image excluding metadata
    display(trainingFolderPos(i).name)
    curImg = imread(strcat('rawTrainingImgs/positive/', trainingFolderPos(i).name)); % current img itteration
    % convert  to greyscale
    greyscaleImg = rgb2gray(curImg);
    processedImg = imresize(greyscaleImg, [20, 20]);  % resize image to 20 x 20
   
    % reshape current image as vector
    processedImg = reshape(processedImg, [1, 400]);
    % store in output matrix
    X(mNeg + i - 3, :) = processedImg;
end

%========== create the labels for the validation set ==========
y = [zeros(mNeg,1); ones(mValidationPos,1)] % add y vector with labels

%========= Preprocessing for validation testing set =========
% prepare negative testing examples
printf('Preparing negative testing examples...\n');
for i=4:mValidationNeg, % for each image excluding metadata
    validationFolderNeg(i).name
    curImgNeg = imread(strcat('validationImgs/negative/', validationFolderNeg(i).name)); % current img itteration
    % convert  to greyscale
    greyscaleImg = rgb2gray(curImgNeg);
    processedImg = imresize(greyscaleImg, [20, 20]);  % resize image to 20 x 20

    % reshape current image as vector
    processedImg = reshape(processedImg, [1, 400])
    processedImg
    % store in output matrix
    valTestSet(i - 3, :) = processedImg;
end

% prepare positve testing examples
printf('Preparing positive testing examples...\n');
for i=4:mValidationPos, % for each image excluding metadata
    display(validationFolderPos(i).name)
    curImgPos = imread(strcat('validationImgs/positive/', validationFolderPos(i).name)); % current img itteration
    % convert  to greyscale
    greyscaleImg = rgb2gray(curImgPos);
    processedImg = imresize(greyscaleImg, [20, 20]);  % resize image to 20 x 20

    % reshape current image as vector
    processedImg = reshape(processedImg, [1, 400]);
    processedImg
    % store in output matrix
    valTestSet(mValidationNeg + i -3, :) = processedImg;
end

%========== create the labels for the validation set ==========
valLabels = [zeros(mValidationNeg,1); ones(mValidationPos,1)]

%========== write to X and y to .mat file as environment variables ==========
save('lemonData.mat', 'X', 'y', 'valTestSet', 'valLabels');
printf('Lemon Image Preprocessing complete.\n');
