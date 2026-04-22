% WWT training and comparing machine learning models

scriptDir = fileparts(mfilename('fullpath'));
load(fullfile(scriptDir, 'data', 'waterTreatmentData.mat'))

% inputs and outputs

inputs = [Q_inf, Q_air_1, Q_air_2, Q_air_3, Q_air_4, Q_air_5, Temp];
outputs = {DO_1, DO_2, DO_3, NO3, NH4};
outputNames = {'DO_1', 'DO_2', 'DO_3', 'NO3', 'NH4'};
algorithmNames = {'Linear Regression', 'Generalized Linear', 'Gaussian Process', ...
    'Support Vector Machine', 'Decision Tree', 'Ensemble', 'Generalized Additive'};
n = size(inputs, 1);

idx = randperm(n);
trainIdx = idx(1:round(0.8*n));
testIdx  = idx(round(0.8*n)+1:end);
originalTrainIdx = trainIdx;
originalTestIdx  = testIdx;
trainInputs = inputs(trainIdx, :);
testInputs  = inputs(testIdx,  :);

rmseResults = zeros(7, 5);
timeResults = zeros(7, 5);

for i = 1:7
    for j = 1:5
        output = outputs{j};
        trainOutput = output(trainIdx);
        testOutput  = output(testIdx);
        [rmseResults(i,j), timeResults(i,j), ~] = trainAlgorithm( ...
            trainInputs, trainOutput, testInputs, testOutput, algorithmNames{i});
    end
    disp(['  Trained: ' algorithmNames{i}]);
end

% error results

disp('RMSE results:');
rmseTable = [table(algorithmNames', 'VariableNames', {'Algorithm'}), ...
             array2table(rmseResults, 'VariableNames', outputNames)];
disp(rmseTable);

disp('Training time results:');
timeTable = [table(algorithmNames', 'VariableNames', {'Algorithm'}), ...
             table(sum(timeResults, 2), 'VariableNames', {'TotalSeconds'})];
disp(timeTable);

% training and testing with different split ratios

splitRatios = [0.5, 0.6, 0.7, 0.8, 0.9];
rmseByRatio = zeros(5, 7);

for r = 1:5
    ratio = splitRatios(r);
    idx = randperm(n);
    splitTrainIdx = idx(1:round(ratio*n));
    splitTestIdx  = idx(round(ratio*n)+1:end);

    splitTrainInputs = inputs(splitTrainIdx, :);
    splitTestInputs  = inputs(splitTestIdx,  :);
    trainOutput = DO_1(splitTrainIdx);
    testOutput  = DO_1(splitTestIdx);

    for i = 1:7
        rmseByRatio(r,i) = trainAlgorithm( ...
            splitTrainInputs, trainOutput, splitTestInputs, testOutput, algorithmNames{i});
    end
    disp(['  Tested: ' num2str(ratio*100) '/' num2str((1-ratio)*100) ' split']);
end

splitLabels = cell(5,1);
for r = 1:5
    splitLabels{r} = [num2str(splitRatios(r)*100) '/' num2str((1-splitRatios(r))*100)];
end
splitTable = [table(splitLabels, 'VariableNames', {'Split'}), ...
              array2table(rmseByRatio, 'VariableNames', algorithmNames)];
disp(splitTable);

trainIdx = originalTrainIdx;
testIdx  = originalTestIdx;

trainMean = mean(inputs(trainIdx, :));
trainStd  = std(inputs(trainIdx,  :));
trainInputsNorm = (inputs(trainIdx, :) - trainMean) ./ trainStd;
testInputsNorm  = (inputs(testIdx,  :) - trainMean) ./ trainStd;

% normalization

rmseNorm = zeros(7, 5);

for i = 1:7
    for j = 1:5
        output = outputs{j};

        outputMean      = mean(output(trainIdx));
        outputStd       = std(output(trainIdx));
        trainOutputNorm = (output(trainIdx) - outputMean) ./ outputStd;
        testOutputNorm  = (output(testIdx)  - outputMean) ./ outputStd;
        testOutputOrig  = output(testIdx);

        [~, ~, trainedModel] = trainAlgorithm( ...
            trainInputsNorm, trainOutputNorm, testInputsNorm, testOutputNorm, algorithmNames{i});

        predOutput    = predict(trainedModel, testInputsNorm) * outputStd + outputMean;
        rmseNorm(i,j) = sqrt(mean((testOutputOrig - predOutput).^2));
    end
    disp(['  Trained (normalized): ' algorithmNames{i}]);
end

origAvg   = mean(rmseResults, 2);
normAvg   = mean(rmseNorm, 2);
pctChange = (normAvg - origAvg) ./ origAvg * 100;
normTable = table(algorithmNames', origAvg, normAvg, pctChange, ...
    'VariableNames', {'Algorithm','Original','Normalized','ChangePct'});
disp(normTable);

% plots

pairs = {
    Q_air_3, 'Q_{air,3}', 'Qair3', '(m^3/day)', DO_1, 'DO_1';
    Q_air_4, 'Q_{air,4}', 'Qair4', '(m^3/day)', DO_2, 'DO_2';
    Q_air_5, 'Q_{air,5}', 'Qair5', '(m^3/day)', DO_3, 'DO_3';
    Q_air_5, 'Q_{air,5}', 'Qair5', '(m^3/day)', NO3,  'NO3';
    Q_air_5, 'Q_{air,5}', 'Qair5', '(m^3/day)', NH4,  'NH4';
    Temp,    'Temp',      'Temp',  '(C)',        DO_3, 'DO_3';
    Temp,    'Temp',      'Temp',  '(C)',        NO3,  'NO3';
    Temp,    'Temp',      'Temp',  '(C)',        NH4,  'NH4';
};

nPairs   = size(pairs, 1);
imageDir = fullfile(scriptDir, 'images');

for i = 1:7
    folderName = strrep(algorithmNames{i}, ' ', '_');
    if ~isfolder(fullfile(imageDir, folderName))
        mkdir(fullfile(imageDir, folderName));
    end
end

for p = 1:nPairs
    inputCol    = pairs{p,1};
    inputLabel  = pairs{p,2};
    inputShort  = pairs{p,3};
    inputUnit   = pairs{p,4};
    outputCol   = pairs{p,5};
    outputLabel = pairs{p,6};

    pairTitle = [inputLabel ' -> ' outputLabel];
    disp(['--- Pair: ' pairTitle ' ---']);

    pairRMSE = zeros(7,1);
    pairTime = zeros(7,1);

    for i = 1:7
        trainInput  = inputCol(originalTrainIdx);
        testInput   = inputCol(originalTestIdx);
        trainOutput = outputCol(originalTrainIdx);
        testOutput  = outputCol(originalTestIdx);

        [pairRMSE(i), pairTime(i), ~] = trainAlgorithm( ...
            trainInput, trainOutput, testInput, testOutput, algorithmNames{i});

        plotWithErrors(inputCol, outputCol, algorithmNames{i}, imageDir, ...
            inputLabel, inputUnit, outputLabel, inputShort, originalTrainIdx, originalTestIdx);
    end

    pairRMSETable = table(algorithmNames', pairRMSE, pairTime, ...
        'VariableNames', {'Algorithm','RMSE','TrainSec'});
    disp(pairRMSETable);

    pairSplitRMSE = zeros(5, 7);
    for r = 1:5
        ratio = splitRatios(r);
        idx = randperm(n);
        splitTrainIdx = idx(1:round(ratio*n));
        splitTestIdx  = idx(round(ratio*n)+1:end);

        for i = 1:7
            pairSplitRMSE(r,i) = trainAlgorithm( ...
                inputCol(splitTrainIdx), outputCol(splitTrainIdx), ...
                inputCol(splitTestIdx),  outputCol(splitTestIdx), algorithmNames{i});
        end
    end

    pairSplitTable = [table(splitLabels, 'VariableNames', {'Split'}), ...
                      array2table(pairSplitRMSE, 'VariableNames', algorithmNames)];
    disp(pairSplitTable);

    inputMean = mean(inputCol(originalTrainIdx));
    inputStd = std(inputCol(originalTrainIdx));
    trainInputNorm = (inputCol(originalTrainIdx) - inputMean) ./ inputStd;
    testInputNorm = (inputCol(originalTestIdx)  - inputMean) ./ inputStd;

    outputMean = mean(outputCol(originalTrainIdx));
    outputStd = std(outputCol(originalTrainIdx));
    trainOutputNorm = (outputCol(originalTrainIdx) - outputMean) ./ outputStd;
    testOutputNorm = (outputCol(originalTestIdx)  - outputMean) ./ outputStd;
    testOutputOrig = outputCol(originalTestIdx);

    pairNormRMSE = zeros(7,1);
    for i = 1:7
        [~, ~, mdl] = trainAlgorithm( ...
            trainInputNorm, trainOutputNorm, testInputNorm, testOutputNorm, algorithmNames{i});
        pred = predict(mdl, testInputNorm) * outputStd + outputMean;
        pairNormRMSE(i) = sqrt(mean((testOutputOrig - pred).^2));
    end

    pctChg = (pairNormRMSE - pairRMSE) ./ pairRMSE * 100;
    pairNormTable = table(algorithmNames', pairRMSE, pairNormRMSE, pctChg, ...
        'VariableNames', {'Algorithm','OriginalRMSE','NormalizedRMSE','ChangePct'});
    disp(pairNormTable);
end

% functions

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

function plotWithErrors(inputCol, outputCol, algoName, imageDir, inputLabel, inputUnit, outputLabel, inputShort, trainIdx, testIdx)
    trainInput = inputCol(trainIdx);
    testInput = inputCol(testIdx);
    trainOutput = outputCol(trainIdx);
    testOutput = outputCol(testIdx);

    [~, ~, trainedModel] = trainAlgorithm( ...
        trainInput, trainOutput, testInput, testOutput, algoName);

    [testSorted, sortIdx] = sort(testInput);
    outputSorted = testOutput(sortIdx);
    predSorted = predict(trainedModel, testSorted);
    rmse = sqrt(mean((outputSorted - predSorted).^2));

    fig = figure('Visible', 'off');
    plot(testSorted, outputSorted, 'o', 'MarkerSize', 3);
    hold on;
    plot(testSorted, predSorted, 'm', 'LineWidth', 2);

    for i = 1:50:length(testSorted)
        plot([testSorted(i) testSorted(i)], [predSorted(i), outputSorted(i)], 'r');
    end

    xlabel([inputLabel ' ' inputUnit]);
    ylabel([outputLabel ' (mg/L)']);
    title([algoName ' — RMSE: ' num2str(round(rmse, 4))]);
    legend('Data', 'Model', 'Error', 'Location', 'best');

    folderName = strrep(algoName, ' ', '_');
    fname = ['singleInput_' inputShort '_' outputLabel '.png'];
    saveas(fig, fullfile(imageDir, folderName, fname));
    close(fig);
end

