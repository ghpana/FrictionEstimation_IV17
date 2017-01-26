%% Support vector machine classifier
% Comment: An SVM model is a representation of the examples as points 
% in space, mapped so that the examples of the separate categories are 
% divided by a clear gap that is as wide as possible. New examples are 
% then mapped into that same space and predicted to belong to a 
% category based on which side of the gap they fall.


%% Import the dataset
testdataset = cleareddataset;

% Possible features sets
WO_NONE = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Prevfriction = [1 2 4 5 6 7 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Dewpoint = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 29 30 31 32 33 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
%WO_Duration = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
%WO_Distance = [1 2 4 5 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Temp = [1 2 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Wiperspeed = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Rain = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 34 35 36 37 38 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Snow = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 39 40 41 42 43 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
W_4hours = [1 2 4 5 6 5 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
W_3hours = [1 2 4 5 6 8 9 10 11 12 13 18 19 20 21 22 23 28 33 38 43 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
W_2hours = [1 2 4 5 6 8 9 10 11 12 13 17 18 19 20 21 22 23 27 28 32 33 37 38 42 43 44 45 46 47 48 52 53 54 55 56 57 58 59 60 61 62 63 64 65];
W_1hours = [1 2 4 5 6 8 9 10 11 12 13 16 17 18 19 20 21 22 23 26 27 28 31 32 33 36 37 38 41 42 43 44 45 46 47 48 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65];
% 42 43
% Default feature set
remove_these = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 29 30 31 32 33 37 38 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65];

remove_these = remove_these;
%% Calculate the new friction value
% Weight the friction measurements w.r.t time and distance
% Max search region (gps metric.)
d_t = 0.04;

% Get the distances from the three last friction measurements
d_1 = testdataset(:,5);
d_2 = testdataset(:,58);
d_2(d_2>d_t) = d_t;
d_3 = testdataset(:,61);
d_3(d_3>d_t) = d_t;

% Get total distance
d_tt = (1-d_1/d_t)+(1-d_2/d_t)+(1-d_3/d_t);

% Max 5 hours
t_t = 5*60;

% And the duration since the last three measurements
t_1 = testdataset(:,6);
t_2 = testdataset(:,59);
t_2(t_2>t_t) = t_t;
t_3 = testdataset(:,62);
t_3(t_3>t_t) = t_t;

ones_ = ones(length(t_1),1);
% Calculate total time
t_tt = (ones_-t_1/t_t)+(ones_-t_2/t_t)+(ones_-t_3/t_t);

% Get the friction values from the three last measurements
% Remove measuremments thats too far or old 
f_1 = testdataset(:,7);
f_2 = testdataset(:,60);
f_2((d_2>d_t) | (t_2 > t_t)) = 0;
f_3 = testdataset(:,63);
f_3((d_3>d_t) | (t_3 > t_t)) = 0;

% Calculate the weighted friction values w.r.t time and distance
f_prev_dis = (f_1.*(ones_-d_1/d_t)+f_2.*(ones_-d_2/d_t)+f_3.*(ones_-d_3/d_t))./d_tt;
f_prev_dur = (f_1.*(ones_-t_1/t_t)+f_2.*(ones_-t_2/t_t)+f_3.*(ones_-t_3/t_t))./t_tt;

% Update the friction measurements by a new weighted friction value
testdataset(:,7) = (f_prev_dis+f_prev_dur)./2;


%%
% Remove these features from the dataset
testdataset(:,remove_these) = [];

X = testdataset(:,2:size(testdataset,2));

% Normalize the input dataset
X_norm = X - (ones(size(X, 1), 1)*mean(X, 1));
X = X_norm./(ones(size(X, 1), 1)*(std(X_norm, 1)));

% Apply PCA to reduce the number of dimensions (default 18)
[COEFF,SCORE] = princomp(X);
X = SCORE(:,1:16);

% Set slippery threshold
Y = sign(testdataset(:,1)-0.35);
Y = -Y;

%Y(Y(:) == -1) = -1
%Y(1:300)=-1

%% Evaluate SVM, Create a confusion matrix
tic

% Set error rate, sensitivity and specificity to zero
error_rate = 0;
sensitivity = 0;
specificity= 0;

% Run cross validation 20 times
loops = 1;
for loop=1:loops
    
    %data partition
    order = unique(Y); % Order of the group labels
    cp = cvpartition(Y,'k',5); %5-folds

    %prediction function
    f = @(xtr,ytr,xte,yte)confusionmat(yte,...
    predict(fitcsvm(xtr, ytr,'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto','ClassNames',[-1,1]),xte),'order',order);

    % missclassification error
    cfMat = crossval(f,X,Y,'partition',cp);
    
    % Get the confusion matrix
    cfMat = reshape(sum(cfMat),2,2);
    error_rate = error_rate + 1-(sum(diag(cfMat)))/sum(sum(cfMat));
    sensitivity = sensitivity + cfMat(2,2)/(cfMat(2,2)+cfMat(2,1));
    specificity = specificity + cfMat(1,1)/(cfMat(1,1)+cfMat(1,2));
end

cfMat

% Show final results
disp('Final Error rate')
error_rate/loops
disp('Final Sensitivity')
sensitivity/loops
disp('Final Specificity')
specificity/loops
toc

