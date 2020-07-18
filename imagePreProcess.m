% ========== Process the raw pngs from the training set folder and write to a .mat file ==========
pkg load image % import forge imag pkg
% load image matrix and label vector
imgFolder = dir('rawTrainingImgs');
validationFolderNeg = dir('validationImgs/negative');
validationFolderPos = dir('validationImgs/postive');

m = length(imgFolder) - 3; % number of images in the dir to process excluding meta data
mValidationNeg = length(validationFolderNeg)
mValidationPos = length(validationFolderPos)
X = zeros(m, 400); % initialize output matrix 400 columns from 20 x 20 images
valTestSet = zeros(mValidationNeg + mValidationPos, 400)

%========= preprocess training set images =========
for i=4:m, % for each image excluding metadata
    display(imgFolder(i).name)
    curImg = imread(strcat('rawTrainingImgs/', imgFolder(i).name)) % current img itteration
    % convert  to greyscale
    greyscaleImg = rgb2gray(curImg);
    processedImg = imresize(greyscaleImg, [20, 20]);  % resize image to 20 x 20
   

    % reshape current image as vector
    processedImg = reshape(processedImg, [1, 400]);
    % store in output matrix
    X(i - 3, :) = processedImg;
end

% add y vector with labels
% because this is a binary classifier we must label these as the positive class "1"
y = ones(m, 1);

%========= Preprocessing for validation testing set =========
% prepare negative testing examples
for i=4:mValidationNeg, % for each image excluding metadata
    display(validationFolder(i).name)
    curImgNeg = imread(strcat('validationImgs/negative', validationFolder(i).name)) % current img itteration
    % convert  to greyscale
    greyscaleImg = rgb2gray(curImgNeg);
    processedImg = imresize(greyscaleImg, [20, 20]);  % resize image to 20 x 20

    % reshape current image as vector
    processedImg = reshape(processedImg, [1, 400])
    processedImg
    % store in output matrix
    valTestSet(i -3, :) = processedImg
end

% prepare positve testing examples
for i=4:mValidationPos, % for each image excluding metadata
    display(validationFolder(i).name)
    curImgPos = imread(strcat('validationImgs/positive', validationFolder(i).name)) % current img itteration
    % convert  to greyscale
    greyscaleImg = rgb2gray(curImgPos);
    processedImg = imresize(greyscaleImg, [20, 20]);  % resize image to 20 x 20

    % reshape current image as vector
    processedImg = reshape(processedImg, [1, 400])
    processedImg
    % store in output matrix
    valTestSet(mValidationNeg + i -3, :) = processedImg
end

%========== create the labels for the validation set ==========
valLabels = [zeros(mValidationNeg,1); ones(mValidationPos,1)]

%========== write to X and y to .mat file as environment variables ==========
save('lemonData.mat', 'X', 'y', 'valTestSet', 'valLabels');
printf('Lemon Image Preprocessing complete.\n');
