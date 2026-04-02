% WWT training and comparing machine learning models

scriptDir = fileparts(mfilename('fullpath'));
load(fullfile(scriptDir, 'data', 'waterTreatmentData.mat'))

inputs = [Q_inf, Q_air_1, Q_air_2, Q_air_3, Q_air_4, Q_air_5, Temp];
outputs = {DO_1, DO_2, DO_3, NO3, NH4};
outputNames = {'DO_1', 'DO_2', 'DO_3', 'NO3', 'NH4'};
algorithmNames = {'Linear Regression', 'Generalized Linear', 'Gaussian Process', ...
    'Support Vector Machine', 'Decision Tree', 'Ensemble', 'Generalized Additive'};
n = size(inputs, 1);

disp('Training...');

% 80/20 split
idx = randperm(n);
trainIdx = idx(1:round(0.8*n));
testIdx = idx(round(0.8*n)+1:end);
trainInputs = inputs(trainIdx, :);
testInputs = inputs(testIdx, :);

rmseResults = zeros(7, 5);
timeResults = zeros(7, 5);

for i = 1:7
    for j = 1:5
        output = outputs{j};
        trainOutput = output(trainIdx);
        testOutput = output(testIdx);
        [rmseResults(i,j), timeResults(i,j), ~] = trainAlgorithm(trainInputs, trainOutput, testInputs, testOutput, algorithmNames{i});
    end
    disp(['Trained: ' algorithmNames{i}]);
end

disp('RMSE results:');
rmseTable = [table(algorithmNames', 'VariableNames', {'Algorithm'}), ...
             array2table(rmseResults, 'VariableNames', outputNames)];
disp(rmseTable);

disp('Elapsed time results:');
timeTable = [table(algorithmNames', 'VariableNames', {'Algorithm'}), ...
             table(sum(timeResults, 2), 'VariableNames', {'TotalSeconds'})];
disp(timeTable);

% Save original 80/20 split indices for normalization section
originalTrainIdx = trainIdx;
originalTestIdx = testIdx;

disp('Split ratios...');

splitRatios = [0.5, 0.6, 0.7, 0.8, 0.9];
rmseByRatio = zeros(5, 7);

for r = 1:5
    ratio = splitRatios(r);
    idx = randperm(n);
    splitTrainIdx = idx(1:round(ratio*n));
    splitTestIdx = idx(round(ratio*n)+1:end);
    
    trainInputs = inputs(splitTrainIdx, :);
    testInputs = inputs(splitTestIdx, :);
    trainOutput = DO_1(splitTrainIdx);
    testOutput = DO_1(splitTestIdx);
    
    for i = 1:7
        rmseByRatio(r,i) = trainAlgorithm(trainInputs, trainOutput, testInputs, testOutput, algorithmNames{i});
    end
    disp(['Tested: ' num2str(ratio*100) '% split']);
end

splitLabels = cell(5, 1);
for r = 1:5
    splitLabels{r} = [num2str(splitRatios(r)*100) '%'];
end
splitTable = [table(splitLabels, 'VariableNames', {'Split'}), ...
              array2table(rmseByRatio, 'VariableNames', algorithmNames)];
disp(splitTable);

disp('Plotting...');

imageDir = fullfile(scriptDir, 'images');

for i = 1:7
    plotWithErrors(Q_inf, DO_1, algorithmNames{i}, imageDir);
end
disp('  Saved 7 plots to images/');

disp('Normalization...');

% Restore original 80/20 split
trainIdx = originalTrainIdx;
testIdx = originalTestIdx;

% Normalize inputs using training data
trainMean = mean(inputs(trainIdx, :));
trainStd = std(inputs(trainIdx, :));
trainInputsNorm = (inputs(trainIdx, :) - trainMean) ./ trainStd;
testInputsNorm = (inputs(testIdx, :) - trainMean) ./ trainStd;

rmseNorm = zeros(7, 5);

for i = 1:7
    for j = 1:5
        output = outputs{j};
        
        outputMean = mean(output(trainIdx));
        outputStd = std(output(trainIdx));
        trainOutputNorm = (output(trainIdx) - outputMean) ./ outputStd;
        testOutputNorm = (output(testIdx) - outputMean) ./ outputStd;
        testOutputOrig = output(testIdx);
        
        [~, ~, trainedModel] = trainAlgorithm(trainInputsNorm, trainOutputNorm, testInputsNorm, testOutputNorm, algorithmNames{i});
        
        % Transform predictions to original scale
        predOutput = predict(trainedModel, testInputsNorm) * outputStd + outputMean;
        
        rmseNorm(i,j) = sqrt(mean((testOutputOrig - predOutput).^2));
    end
    disp(['Trained (normalized): ' algorithmNames{i}]);
end

originalRmse = mean(rmseResults, 2);
normalizedRmse = mean(rmseNorm, 2);
percentChange = (normalizedRmse - originalRmse) ./ originalRmse * 100;
normTable = table(algorithmNames', originalRmse, normalizedRmse, percentChange, ...
    'VariableNames', {'Algorithm', 'Original', 'Normalized', 'ChangePct'});
disp(normTable);


function [rmse, elapsed, model] = trainAlgorithm(trainInputs, trainOutput, testInputs, testOutput, algoName)
    tic;
    
    if strcmp(algoName, 'Linear Regression')
        model = fitlm(trainInputs, trainOutput);
    elseif strcmp(algoName, 'Generalized Linear')
        model = fitglm(trainInputs, trainOutput);
    elseif strcmp(algoName, 'Gaussian Process')
        model = fitrgp(trainInputs, trainOutput);
    elseif strcmp(algoName, 'Support Vector Machine')
        model = fitrsvm(trainInputs, trainOutput);
    elseif strcmp(algoName, 'Decision Tree')
        model = fitrtree(trainInputs, trainOutput);
    elseif strcmp(algoName, 'Ensemble')
        model = fitrensemble(trainInputs, trainOutput);
    elseif strcmp(algoName, 'Generalized Additive')
        model = fitrgam(trainInputs, trainOutput);
    end
    
    elapsed = toc;
    predOutput = predict(model, testInputs);
    rmse = sqrt(mean((testOutput - predOutput).^2));
end

function plotWithErrors(inputCol, outputCol, algoName, saveDir)
    [inputSorted, sortIdx] = sort(inputCol);
    outputSorted = outputCol(sortIdx);
    
    [~, ~, trainedModel] = trainAlgorithm(inputCol, outputCol, inputCol, outputCol, algoName);
    
    predOutput = predict(trainedModel, inputSorted);
    mse = mean((outputSorted - predOutput).^2);
    
    fig = figure('Visible', 'off');
    plot(inputSorted, outputSorted, 'o', 'MarkerSize', 3);
    hold on;
    plot(inputSorted, predOutput, 'm', 'LineWidth', 2);
    
    for i = 1:50:length(inputSorted)
        plot([inputSorted(i) inputSorted(i)], [predOutput(i), outputSorted(i)], 'r');
    end
    
    xlabel('Q_{inf} (m^3/day)');
    ylabel('DO_1 (mg/L)');
    title([algoName ' - MSE: ' num2str(round(mse, 4))]);
    legend('Data', 'Model', 'Error', 'Location', 'best');
    
    saveas(fig, fullfile(saveDir, ['singleInput_' algoName '.png']));
    close(fig);
end
