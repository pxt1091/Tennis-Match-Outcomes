clear all; clc;
load('dataset.mat');

X = [Surface, P1Rank, P2Rank, RankDelta, P1Pts, P2Pts, PtsDelta, ScoreDelta, Sets, Coeff, P1Prob, P2Prob];
y = Player1Win;

[Xnorm mu sigma] = featureNormalize(X);

Xtrain = Xnorm(1:15576,:);
Xtest = Xnorm(15577:end,:);
Ytrain = y(1:15576,:);
Ytest = y(15577:end,:);

% Add intercept term to X
Xdata = [ones(length(Xtrain),1) Xtrain];

theta = ((Xdata'*Xdata)\Xdata')*Ytrain;
lambda = 1;

Xdatatest = [ones(length(Xtest),1) Xtest];
ypred = (Xdatatest * theta) > 0.5;

diff = (Ytest - ypred).^2;
error = sum(diff);
accuracy = 100 - (error*100/length(Ytest))

%Md1 = fitrsvm(Xtrain,Ytrain);
%ypred_svm = predict(Md1, Xtest);
%ypred_svm_final = ypred_svm > 0.5;

%diff1 = (Ytest - ypred_svm_final).^2;
%error1 = sum(diff1);
%accuracy = 100 - (error1*100/length(Ytest))

%Md2 = fitrsvm(Xtrain,Ytrain);
%ypred_svm = predict(Md2, Xtest);
%ypred_svm_final = ypred_svm > 0.5;

%diff2 = (Ytest - ypred_svm_final).^2;
%error2 = sum(diff2);
%accuracy = 100 - (error2*100/length(Ytest))

Md3 = fitcnb(Xtrain,Ytrain);
ypred_3 = predict(Md3, Xtest);
ypred_3_final = ypred_3 > 0.5;

diff3 = (Ytest - ypred_3_final).^2;
error3 = sum(diff3);
accuracy = 100 - (error3*100/length(Ytest))


