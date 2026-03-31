load all_data.csv
load energyDataset.mat
load waterTreatmentData.mat

% input variables 
X = [Q_inf, Q_air_1, Q_air_2, Q_air_3, Q_air_4, Q_air_5, Temp];

% output variables 
outputs = {DO_1, DO_2, DO_3, NO3, NH4};
outputNames = {'DO_1', 'DO_2', 'DO_3', 'NO3', 'NH4'};

% training and testing split
n = size(X, 1);
idx = randperm(n);
trainIdx = idx(1:round(0.8*n));
testIdx = idx(round(0.8*n)+1:end);

X_train = X(trainIdx, :);
X_test = X(testIdx, :);

% linear regression
for i = 1:5
    y = outputs{i};
    y_train = y(trainIdx);
    y_test = y(testIdx);

    mdl = fitlm(X_train, y_train);
    y_pred = predict(mdl, X_test);
    rmse_val = sqrt(mean((y_test - y_pred).^2));
    fprintf('Linear Reg - %s RMSE: %.4f\n', outputNames{i}, rmse_val);
end

% generalized linear regression
for i = 1:5
    y = outputs{i};
    y_train = y(trainIdx);
    y_test = y(testIdx);

    mdl = fitglm(X_train, y_train);
    y_pred = predict(mdl, X_test);
    rmse_val = sqrt(mean((y_test - y_pred).^2));
    fprintf('Generalized Linear Reg - %s RMSE: %.4f\n', outputNames{i}, rmse_val);
end

% gaussian process regression
for i = 1:5
    y = outputs{i};
    y_train = y(trainIdx);
    y_test = y(testIdx);

    mdl = fitrgp(X_train, y_train);
    y_pred = predict(mdl, X_test);
    rmse_val = sqrt(mean((y_test - y_pred).^2));
    fprintf('Gaussian Process Reg - %s RMSE: %.4f\n', outputNames{i}, rmse_val);
end

% support vector machine
for i = 1:5
    y = outputs{i};
    y_train = y(trainIdx);
    y_test = y(testIdx);

    mdl = fitrsvm(X_train, y_train);
    y_pred = predict(mdl, X_test);
    rmse_val = sqrt(mean((y_test - y_pred).^2));
    fprintf('Support Vector Machine - %s RMSE: %.4f\n', outputNames{i}, rmse_val);
end

% decision tree
for i = 1:5
    y = outputs{i};
    y_train = y(trainIdx);
    y_test = y(testIdx);

    mdl = fitrtree(X_train, y_train);
    y_pred = predict(mdl, X_test);
    rmse_val = sqrt(mean((y_test - y_pred).^2));
    fprintf('Decision Tree - %s RMSE: %.4f\n', outputNames{i}, rmse_val);
end

% ensemble of learners
for i = 1:5
    y = outputs{i};
    y_train = y(trainIdx);
    y_test = y(testIdx);

    mdl = fitrensemble(X_train, y_train);
    y_pred = predict(mdl, X_test);
    rmse_val = sqrt(mean((y_test - y_pred).^2));
    fprintf('Ensemble of Learners - %s RMSE: %.4f\n', outputNames{i}, rmse_val);
end

% generalized additive model
for i = 1:5
    y = outputs{i};
    y_train = y(trainIdx);
    y_test = y(testIdx);

    mdl = fitrgam(X_train, y_train);
    y_pred = predict(mdl, X_test);
    rmse_val = sqrt(mean((y_test - y_pred).^2));
    fprintf('Generalized Additive Model - %s RMSE: %.4f\n', outputNames{i}, rmse_val);
end