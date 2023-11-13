% ================================================================
%                       Machine Learning                   
%
%                             Lab 2
% 
% =================================================================
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

%% ===================== Preparation ======================================
% Download lab2.zip from Blackboard and unzip it to  
% somewhere on your computer. Change the path using the command cd to the 
% folder where your files are located. 
% ====================== YOUR CODE HERE ===================================
cd(...
% =========================================================================

%% =========== Part 1: Dimensionality reduction =========================
clear all;

% PCA, LDA and t-SNE are very useful tools for plotting data that has more than 2
% dimensions. They can also be used for feature selection. 
% We will use the smallMNIST dataset to demonstrate how they work.
% This data set is used to classify handwritten digits 0-9. The
% file smallMNIST contains a smaller version of the popular MNIST data set.
% The variable X contains 5000 images (500 per digit). Each image is 20x20
% pixels big, so the total number of features per image is 400. The
% variable y is the labelvector where y(i) = k means instance i belongs to
% digit k except for k = 10 which represents the digit 0. 
load('smallMNIST.mat'); % Loads X and y

% Let's reduce the number of images to visualize the data better.
X = X(1:10:end,:);
y = y(1:10:end);

% ====================== YOUR CODE HERE ======================
% Use the function imagesc to plot the first image in this data set X(1,:).
% You can use the function reshape to reshape this vector of 400 elements
% to a 20 x 20 matrix.

% ============================================================

% First we can normalize the data with mean normalization. We can use the
% matlab function zscore which does the same thing as featureMeanNormalize.m 
% from lab1. 
X = zscore(X);

% Use the functions pca, lda, and tsne to plot calculate the reduced
% feature space and plot for each algorithm the data points in different
% colors. To get help of a certain function, type: 
% >> help pca

% ====================== YOUR CODE HERE ===============================
figure;
% Now we use the matlab pca function. Notice how PCA is an unsupervised
% algorithm and does not use the label vector y. Plot the projected data in 
% two dimensions. We only use the first two columns in Xpca


% Next we use LDA on the same data. We will use an implementation of LDA 
% from the Matlab Toolbox for Dimensionality Reduction. Notice how lda also 
% takes the label vector y as input.


% Next we use LDA on the same data. We will use an implementation of LDA 
% from the Matlab Toolbox for Dimensionality Reduction. Notice how lda also 
% takes the label vector y as input.


% =====================================================================


%% ============= Part 2: Clustering ================================
clear all;

% We will use MATLABs GMM implementation for this part. First we load the
% data.
load simplecluster
X = zscore(X);

% Next we fit a GMM to the data
numberOfClusters = 3;
obj = gmdistribution.fit(X, numberOfClusters);

% Then classify each data point into one of the clusters
idx = cluster(obj,X);

% Plot the data and display the cluster number in each cluster center
plotData(X, idx); title(['GMM, #Clusters = ' num2str(numberOfClusters)]);

% ====================== YOUR CODE HERE ===================================
% Now do the same for k-means using the function kmeans. Type >> help kmeans
% for more information how this function works. 


% =========================================================================

%% ========================== Part 3: Classification ======================
clear all;
load smallMNIST; % load X and y

% ====================== YOUR CODE HERE ===================================
% You can use a PCA to reduce the number of dimensions of the data here
%N = 10; % number of components to use
%[coefs, Xpca, variances] = pca(X);
%X = Xpca(:,1:N);
% =========================================================================

% We randomly split the data X and label vector y into 70% training
% data that we will use to train the classifiers and 30% testing data that
% we use to validate the model. Use the same random seed for X and y.
seed = 1;
[trainX, testX] = splitData(X, [0.7; 0.3], seed);
[trainy, testy] = splitData(y, [0.7; 0.3], seed);

% ====================== YOUR CODE HERE ===================================
% Use the functions fitctree, fitcknn, and fitcnb to train a Decision Tree, 
% k-NN classifier, and a Naive Bayes classifier on the training data and 
% training labels trainX and trainy. Calculate the predictions using the 
% function predict on the test data testX and compute the classification 
% accuracy using the test labels testy. 
% HINT: The classfication accuracy can be computed by mean(y_pred(:)==y(:))


% =========================================================================


%% ========= Part 4: Logistic Regression for multiple classes =============
% In this part we will train a logistic regression classifier for the task
% of classifying handwritten digits [0-9]. 
clear all;

% First we load the data from the file smallMNIST.mat which is a reduced 
% set of the MNIST handwritten digit dataset. The full data set can be
% downloaded from http://yann.lecun.com/exdb/mnist/. Our data X consist of 
% 5000 examples of 20x20 images of digits between 0 and 9. The number "0" 
% has the label 10 in the label vector y. The data is already normalized.
load('smallMNIST.mat'); % Gives X, y

% Now we divide the data X and label vector y into training, validation and
% test set. We use the same seed so that we dont get different
% randomizations. We will use hold-out cross validation to select the
% hyperparameter lambda.
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6; 0.3; 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6; 0.3; 0.1], seed);

% Now we train 10 different one vs all logistic regressors. Complete the
% code in trainLogisticReg.m before continuing. 
lambda = 0;
all_theta = trainLogisticReg(Xtrain, ytrain, lambda);

% Now we calculate the predictions using all 10 models. 
ypredtrain = predictLogisticReg(all_theta, Xtrain);
ypredval = predictLogisticReg(all_theta, Xval);
ypredtest = predictLogisticReg(all_theta, Xtest);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);


%% ========= Part 5: Regularization for Logistic Regression =============
% First we will add regularization to our previously written logistic
% regression. Copy costLogisticRegression.m and put it in the same folder 
% as this file.

clear all;

% Implement L2 weight decay regularization in costLogisticRegression.m. 
% Do not regularize the first element in theta. Check the gradients on a
% small randomized test data.
X = randn(10,10);
y = randi(2,10,1)-1;
initial_theta = randn(10,1);
lambda = 1;
[J, grad] = costLogisticRegression(initial_theta, X, y, lambda);
numgrad = checkGradient(@(p) costLogisticRegression(p, X, y, lambda), initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% ====================== YOUR CODE HERE ====================
% Calculate the classification accuracy on the train, val, and test sets
% using lambda = 0.1 and print the results

% ==========================================================

%% ================== Part 6: Bias-variance analysis ==============

% ====================== YOUR CODE HERE ======================
% Hint: You can use set(gca,'Xtick', 1:length(lambda_list), 'Xticklabel', lambda_list)
% to set the x-label as the values for lambda

% ============================================================

%% ================ Exercises for Pass with distinction (VG) ===============

%% ====== Principal Component Analysis (PCA): Reconstruction =====
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.
clear all;

% Load the data
load('smallMNIST.mat');

% Finish the code in mypca to calculate Xpca before continuing.
[coefs, Xpca, variances] = mypca(X);

% ====================== YOUR CODE HERE ===================================
% We can use Xpca and coefs to reconstruct an approximation of the data back
% to the original space using the top K eigenvectors in coefs. For the i-th 
% example Xpca(i,:), the (approximate) recovered data for dimension j is 
% given as: 
%                 Xrec(i,j) = Xpca(i, 1:K) * coefs(j, 1:K)';
%
% Xrec should have the same size as X.

K = 2;   Xrec = Xpca(:, 1:K) * coefs(:, 1:K)'; subplot(1,3,1); imagesc(reshape(Xrec(1,:),20,20))
K = 50;  Xrec = Xpca(:, 1:K) * coefs(:, 1:K)'; subplot(1,3,2); imagesc(reshape(Xrec(1,:),20,20))
K = 400; Xrec = Xpca(:, 1:K) * coefs(:, 1:K)'; subplot(1,3,3); imagesc(reshape(Xrec(1,:),20,20))

% iter = 1;
% for i = 1:500:5000
%     figure;
%     for K = [2 50 400]
%         for j=1:size(coefs,1)
%             Xrec(i,j) = Xpca(i, 1:K) * coefs(j, 1:K)';
%         end
%         subplot(1,3,iter); imagesc(reshape(Xrec(i,:),20,20))
%         iter = iter + 1;
%     end
%    iter = 1;
% end


% =========================================================================


%% ==================== Implementation of k-means =========================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% ====================== YOUR CODE HERE ===================



% =========================================================

%% ======= Plot original images of MNIST from k-means clustering ==========

% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

clear all;
load('smallMNIST.mat'); % Loads X and y

% ====================== YOUR CODE HERE ===================================
% Use K-means on the smallMNIST data set using K=10. Make a plot with 10 
% original images from each cluster. 

% =========================================================================

%% ============ Plot Learning curves =====================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.




%% ================== (optional) Spam Email Classification ==================
% NOTE: THIS PART IS OPTIONAL

%  Now we will classify emails into Spam or Non-Spam.
clear all;

% The file spamTrain.mat contains the feature vector for 4000 emails stored
% in a matrix X and the labels y. The file spamTest.mat contains 1000
% emails that we will use as test set. 
load('spamTrain.mat'); % Gives X, y to your workspace
load('spamTest.mat'); % % Gives Xtest, ytest to your workspace

% ====================== YOUR CODE HERE ======================
% Use the function fitcsvm to train a Support Vector Machine (SVM) and
% calulate the classification accury on the test set.

for C = [0.001 0.01 0.1 1 10]
    svmmodel = fitcsvm(X, y, 'BoxConstraint', C);
    ypred = predict(svmmodel, Xtest);
    fprintf('%0.2f\t%0.5f\n', C, mean(ypred(:) == ytest(:)))
end


% ============================================================

% OPTIONAL:
% If you want to try your classifier on one of your own email you can copy 
% the email to emailSample.txt and convert the text to a feature vector
% using the code below. I put in one of my academic spam emails as default.
% Then feed the feature vector to your trained classifier. 

% Uncomment to run
file_contents = readFile('emailSample.txt');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
predict(svmmodel, features)




