% ================================================================
%                       Machine Learning                   
%
%                             Lab 3
% 
% ================================================================
% Instructions:
% Follow the code and the comments in this file carefully. You will need to 
% change parts of the code in this file and in other files. The places
% where you need to enter your code is indicated by:
% ====================== YOUR CODE HERE ======================
% ...
% ============================================================
%
% A written report should be handed in before the end of the course.
% The report is graded with either PASS or PASS WITH DISTINCTION. At the 
% end of each cell of code (a cell is separated by %%) there are
% instructions for what to include in the report, e.g.:
% ====================== REPORT ==============================
% Some instructions...
% FOR PASS WITH DISTINCTION: Some more instructions...
% ============================================================
% All the report blocks that contain FOR PASS WITH DISTINCTION need 
% to be completed in order to receive a PASS WITH DISTINCTION on the report.
% 
% MATLAB Tips:
% You can read about any MATLAB function by using the help function, e.g.:
% >> help plot
% To run a single line of code simply highlight the code you want to run 
% and press F9. To run one cell of code first click inside the cell and 
% then press CTRL + ENTER.

%% ===================== Preparation ===========================
% Download lab3.zip from Blackboard and unzip it to  
% somewhere on your computer. Change the path using the command cd to the
% folder where your files are located.
% ====================== YOUR CODE HERE ======================
cd('...')
% ============================================================

addpath('./minFunc/'); %add minFunc to working directory after running the cd command 

%% ============== Part 1: Implementing Neural network ====================
% Time to implement a non-regularized neural network with one hidden layer.
clear  all;

% Create a small randomized data matrix and labelvector for testing your 
% implementation.
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. We start with coding the NN without any
% regularization
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % Weight decay penalty parameter (not used in this exercise)
parameters.beta = 0; % sparsity penalty parameter (not used in this exercise)

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 10 output units. 
[theta thetaSize] = initNNParameters(8, 5, 10);

% Calculate cost and grad and check gradients. Finish the code in 
% costNeuralNetwork.m first.
[cost,grad] = costNeuralNetwork(theta, thetaSize, X, y, parameters);
numGrad = checkGradient( @(p) costNeuralNetwork(p, thetaSize, X, y, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9


%% ==== Part 2: Neural network for handwritten digit classification ======
clear all;

% Load the data set and split into train, val, and test sets.
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters.
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % This is a tunable hyperparameter (not used in this exercise)
parameters.beta = 0; % This is a tunable hyperparameter (not used in this exercise)
numhid = 50; % % This is a tunable hyperparameter

% Initiliaze the network parameters.
numvis = size(X, 1);
numout = length(unique(y));
[theta, thetaSize] = initNNParameters(numvis, numhid, numout);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costNeuralNetwork(p, thetaSize, Xtrain, ytrain, parameters);

% Now, costFunction is a function that takes in only one argument (the 
% neural network parameters). Use tic and toc to see how long the training
% takes.
tic
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
[optTheta, optCost] = minFunc(costFunction, theta, options);
toc

% fmincg takes longer to train. Uncomment if you want to try it.
% tic
% options = optimset('MaxIter', 400, 'display', 'on');
% [optTheta, optCost] = fmincg(costFunction, theta, options);
% toc

% You can visualize what the network has learned by plotting the weights of
% W1 using displayData.
[W1, W2, b1, b2] = theta2params(optTheta, thetaSize);
%displayData(W1);

% Now we predict all three sets.
ypredtrain = predictNeuralNetwork(optTheta, thetaSize, Xtrain);
ypredval = predictNeuralNetwork(optTheta, thetaSize, Xval);
ypredtest = predictNeuralNetwork(optTheta, thetaSize, Xtest);

fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100);
fprintf('Val Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);



%% ============== Part 3: Implementing Auto-encoder =======================
% Time to implement a non-regularized auto-encoder.
clear  all;

% Create a small randomized data matrix and labelvector
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % Weight decay penalty parameter (not used in this exercise)
parameters.beta = 0; % sparsity penalty parameter (not used in this exercise)

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 8 output units (same as the number of input
% units).
[theta, thetaSize] = initAEParameters(8, 5);

% Calculate cost and grad and check gradients. Note how costAutoencoder.m 
% does not require the label vector y.
[cost,grad] = costAutoencoder(theta, thetaSize, X, parameters);
numGrad = checkGradient( @(p) costAutoencoder(p, thetaSize, X, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9


%% ======= Part 4: Reconstructing with Auto-encoder ===================
clear all;

load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % This is a tunable hyperparameter (not used in this exercise)
parameters.beta = 0; % This is a tunable hyperparameter (not used in this exercise)
numhid = 100; % This is a tunable hyperparameter
maxIter = 100; % This is a tunable hyperparameter

% Initiliaze the network parameters. Here we use initAEParameters.m
% instead.
numvis = size(X, 1);
[theta, thetaSize] = initAEParameters(numvis, numhid);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costAutoencoder(p, thetaSize, Xtrain, parameters);

% Now, costFunction is a function that takes in only one argument (the 
% neural network parameters). Use tic and toc to see how long the training
% takes.
tic
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', maxIter);
[optTheta, optCost] = minFunc(costFunction, theta, options);
toc

% fmincg takes longer to train. Uncomment if you want to try it.
% tic
% options = optimset('MaxIter', 400, 'display', 'on');
% [optTheta, optCost] = fmincg(costFunction, theta, options);
% toc

[W1, W2, b1, b2] = theta2params(optTheta, thetaSize);

figure;
h = sigmoid(bsxfun(@plus, W1*Xtrain, b1)); %hidden layer
Xrec = sigmoid(bsxfun(@plus, W2*h, b2)); % reconstruction layer
subplot(1,2,1); displayData(Xtrain(:,1:100)'); title('Original input')
subplot(1,2,2); displayData(Xrec(:,1:100)'); title('Reconstructions')



%% =============== Exercises for pass with distinction (VG) ===============


%% Part 5: Bias-Variance Analysis on the Number of Hidden Units in a Neural Network
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.



%% ========= Part 6: Softmax classification for multiple classes ==========
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% In this part we will train a softmax classifier for the task of 
% classifying handwritten digits [0-9]. 
clear  all;

% Load the same data set. In softmax and neural networks the convention is 
% to let each column be one training input instead of each row as we have 
% previously used. 
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';

% Split into train, val, and test sets
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], seed);

% ======= Part 6a: Implement the softmax classifier ===================

% Initialize theta
numClasses = 10; % Number of classes
initial_theta = reshape(0.005 * randn(numClasses, size(X,1)), [], 1);

% For debugging purposes create a small randomized data matrix and
% labelvector. Calculate cost and grad and check gradients. Finish the code 
% in costSoftmax.m first. If your gradients don't match at first, try setting 
% lambda = 0; to see if the problem is with the error term or the 
% regularization term.
[cost,grad] = costSoftmax(initial_theta, Xtrain(:,1:12), ytrain(1:12), numClasses);
numGrad = checkGradient( @(p) costSoftmax(p, Xtrain(:,1:12), ytrain(1:12), numClasses), initial_theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

% ======= Part 6b: Classification with softmax on raw MNIST data ===================
% If the diff is low, continue to train the softmax classifier on the smallMNIST data.

max_iter = 400; % You can change here
initial_theta = reshape(0.005 * randn(numClasses, size(X,1)), [], 1);
options = struct('display', 'off', 'Method', 'lbfgs', 'maxIter', max_iter);
theta = minFunc( @(p) costSoftmax(p, Xtrain, ytrain, numClasses), initial_theta, options);

% Now calculate the predictions.
ypredtrain = predictSoftmax(theta, Xtrain, numClasses);
ypredval = predictSoftmax(theta, Xval, numClasses);
ypredtest = predictSoftmax(theta, Xtest, numClasses);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);


% == Part 6c: Classification with softmax on hidden layer from Auto-encoder =====
% Use the train an auto-encoder and use the hidden layer activations as
% input to train a softmax classifier.

maxIter = 400;
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % This is a tunable hyperparameter (not used in this exercise)
parameters.beta = 0; % This is a tunable hyperparameter (not used in this exercise)
numvis = size(X,1);
numhid = 100; % You can change here
[theta, thetaSize] = initAEParameters(numvis, numhid);
costFunction = @(p) costAutoencoder(p, thetaSize, Xtrain, parameters);
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', maxIter);
[optTheta, optCost] = minFunc(costFunction, theta, options);

% Feedforward the AE to get the hidden layer
numClasses = 10;
[W1, W2, b1, b2] = theta2params(optTheta, thetaSize);
htrain = sigmoid(bsxfun(@plus, W1*Xtrain, b1)); %hidden layer
hval = sigmoid(bsxfun(@plus, W1*Xval, b1)); %hidden layer
htest = sigmoid(bsxfun(@plus, W1*Xtest, b1)); %hidden layer

% Train a softmax on the hidden layer
max_iter = 400; % You can change here
initial_theta = reshape(0.005 * randn(numClasses, size(htrain,1)), [], 1);
options = struct('display', 'off', 'Method', 'lbfgs', 'maxIter', max_iter);
theta = minFunc( @(p) costSoftmax(p, htrain, ytrain, numClasses), initial_theta, options);

% Now calculate the predictions.
ypredtrain = predictSoftmax(theta, htrain, numClasses);
ypredval = predictSoftmax(theta, hval, numClasses);
ypredtest = predictSoftmax(theta, htest, numClasses);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

% ============================================================



%% ======= Part 7: Using the Deep Learning Toolbox ===================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% Familiarize yourself with the deep learning toolbox in Matlab, see:
% https://se.mathworks.com/help/deeplearning/index.html?s_tid=srchtitle_site_search_1_deep%20learning%20toolbox
%
% Create a simple 1-layer neural network with sigmoid activation function
% in the middle layer and a softmax layer in the output layer. 
% 
% You can follow the guide here:
% https://se.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html

% (optional) Use transfer learning to load a pre-trained deep network (e.g. googlenet or alexnet) and test it on the
% smallMNIST.mat dataset. You need to change the structure of the network (Tips: use the Deep Network Designer)








