%% Logistic regression with k-fold cross validation
% Comment: Logistic regression is a technique in statistics to build a 
% model that is aimed at finding the best fitting and most accurate and 
% sensible model to assess the relationship between a set of responsive 
% variables and at least one explanatory variable.

%% Fetch new dataset
testdataset = cleareddataset;

% Remove these features
WO_NONE = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Prevfriction = [1 2 4 5 6 7 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Dewpoint = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 29 30 31 32 33 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
%WO_Duration = [1 2 4 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
%WO_Distance = [1 2 4 5 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Temp = [1 2 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Wiperspeed = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Rain = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 34 35 36 37 38 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
WO_Snow = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 39 40 41 42 43 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
W_4hours = [1 2 4 5 6 5 6 8 9 10 11 12 13 19 20 21 22 23 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64 65];
W_3hours = [1 2 4 5 6 8 9 10 11 12 13 18 19 20 21 22 23 28 33 38 43 44 45 46 47 48 53 54 55 56 57 58 59 60 61 62 63 64];
W_2hours = [1 2 4 5 6 8 9 10 11 12 13 17 18 19 20 21 22 23 27 28 32 33 37 38 42 43 44 45 46 47 48 52 53 54 55 56 57 58 59 60 61 62 63 64 65];
W_1hours = [1 2 4 5 6 8 9 10 11 12 13 16 17 18 19 20 21 22 23 26 27 28 31 32 33 36 37 38 41 42 43 44 45 46 47 48 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65];

remove_these_default = [1 2 4 5 6 8 9 10 11 12 13 19 20 21 22 23 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65];

remove_these_default = remove_these_default;

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
testdataset(:,remove_these_default) = [];

% Get the features
X = testdataset(:,2:size(testdataset,2))
mean(X)

%%
Y = sign(testdataset(:,1)-0.35)
Y = -Y
Y(Y(:) == -1) = 0

% Normalize the input dataset
X_norm = X - (ones(size(X, 1), 1)*mean(X, 1));
X = X_norm./(ones(size(X, 1), 1)*(std(X_norm, 1)));


%% PCA, reduce the dimensions into 14
[COEFF,SCORE] = princomp(X)
X = SCORE(:,1:14)


%% Partition data into 5 folds
K = 5;
cv = cvpartition(numel(Y), 'kfold',K);
confusion_matrix = zeros(2,2);
min_test_batch_size = round(numel(Y)/K)-1;

% Mean square error and area under curve
%mse = zeros(K,1);
%AUC = zeros(K,1);

% For ROC plot
%Xb = zeros(min_test_batch_size,1)
%Yb = zeros(min_test_batch_size,1)

% Set error rate, sensitivity and specificity to zero
error_rate = 0;
sensitivity = 0;
specificity = 0;

% Run cross validation 30 times
loops = 1
for loop = 1:loops
    for k=1:K
        % training/testing indices for this fold
        trainIdx = cv.training(k);
        testIdx = cv.test(k);

        % train GLM model
        mdl = fitglm(X(trainIdx,:), Y(trainIdx), ...
            'linear', 'Distribution','binomial','link','logit');
        
        % Alt. version for older versions of matlab
        %mdl = GeneralizedLinearModel.fit(X(trainIdx,:), Y(trainIdx), ...
        %    'linear', 'Distribution','binomial','link','logit');

        % Get the prediction from the regression model
        Y_hat = predict(mdl, X(testIdx,:));

        % For ROC
        %scores = mdl.Fitted.Probability;
        %[Xb_temp,Yb_temp,~,AUC(k)] = perfcurve(Y(testIdx,:),Y_hat,1)
        %Xb(1:min_test_batch_size) = Xb(1:min_test_batch_size)+Xb_temp(1:min_test_batch_size);
        %Yb(1:min_test_batch_size) = Yb(1:min_test_batch_size)+Yb_temp(1:min_test_batch_size);

        % compute mean squared error
        mse(k) = mean((Y(testIdx) - Y_hat).^2);
        Y_hat = sign(Y_hat - 0.35);
        Y_hat(Y_hat(:) == -1) = 0;

        confusion_matrix(1,1) = confusion_matrix(1,1) + sum(Y(testIdx,:) == 0 & Y_hat == 0);
        confusion_matrix(1,2) = confusion_matrix(1,2) + sum(Y(testIdx,:) == 0 & Y_hat == 1);
        confusion_matrix(2,1) = confusion_matrix(2,1) + sum(Y(testIdx,:) == 1 & Y_hat == 0);
        confusion_matrix(2,2) = confusion_matrix(2,2) + sum(Y(testIdx,:) == 1 & Y_hat == 1);
    end

    % Get mean
    %Xb = Xb/K;
    %Yb = Yb/K;

    % plot ROC
    %plot(Xb,Yb,'-b')
    %title('ROC')
    % xlabel('False positive rate')
    % ylabel('True positive rate')

    %Area under curve
    %disp('Area under curve')
    %mean(AUC)

    % Result Error rate
    error_rate = error_rate + 1-(confusion_matrix(1,1)+confusion_matrix(2,2))/sum(sum(confusion_matrix))
    %Sensitivity
    sensitivity = sensitivity + confusion_matrix(2,2)/(confusion_matrix(2,2)+confusion_matrix(2,1))
    %Specificity
    specificity = specificity + confusion_matrix(1,1)/(confusion_matrix(1,1)+confusion_matrix(1,2))
    
    % average RMSE across k-folds
    %avrg_rmse = mean(sqrt(mse));
end

% Show results
disp('Final Error rate')
error_rate/loops
disp('Final Sensitivity')
sensitivity/loops
disp('Final Specificity')
specificity/loops

