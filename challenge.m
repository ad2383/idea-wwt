
scriptDir = fileparts(mfilename('fullpath'));
load(fullfile(scriptDir, 'data', 'waterTreatmentData.mat'))

algorithmNames = {'Linear Regression', 'Generalized Linear', 'Gaussian Process', ...
    'Support Vector Machine', 'Decision Tree', 'Ensemble', 'Generalized Additive'};

input1    = Q_air_3;
input2    = Q_air_4;
output    = DO_1;
input1Label = 'Q_{air,3} (m^3/day)';
input2Label = 'Q_{air,4} (m^3/day)';
outputLabel = 'DO_1 (mg/L)';

n = length(output);
idx = randperm(n);
trainIdx = idx(1:round(0.8*n));
testIdx  = idx(round(0.8*n)+1:end);

trainInputs = [input1(trainIdx), input2(trainIdx)];
testInputs  = [input1(testIdx),  input2(testIdx)];
trainOutput = output(trainIdx);
testOutput  = output(testIdx);

imageDir = fullfile(scriptDir, 'images', '3D');
if ~isfolder(imageDir), mkdir(imageDir); end

rmseResults = zeros(7,1);
timeResults = zeros(7,1);

for i = 1:7
    [rmseResults(i), timeResults(i), ~] = trainAlgorithm( ...
        trainInputs, trainOutput, testInputs, testOutput, algorithmNames{i});
end

resultsTable = table(algorithmNames', rmseResults, timeResults, ...
    'VariableNames', {'Algorithm','RMSE','TrainSec'});
disp('3D Model Results (Q_air_3 + Q_air_4 -> DO_1):');
disp(resultsTable);

for i = 1:7
    plot3DModel(input1, input2, output, algorithmNames{i}, ...
        input1Label, input2Label, outputLabel, imageDir, trainIdx, testIdx);
    disp(['  Plotted: ' algorithmNames{i}]);
end

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

function plot3DModel(input1, input2, output, algoName, ...
        input1Label, input2Label, outputLabel, imageDir, trainIdx, testIdx)

    trainInputs = [input1(trainIdx), input2(trainIdx)];
    testInputs  = [input1(testIdx),  input2(testIdx)];
    trainOutput = output(trainIdx);
    testOutput  = output(testIdx);

    [~, ~, model] = trainAlgorithm( ...
        trainInputs, trainOutput, testInputs, testOutput, algoName);

    x1range = linspace(min(testInputs(:,1)), max(testInputs(:,1)), 50);
    x2range = linspace(min(testInputs(:,2)), max(testInputs(:,2)), 50);
    [X1, X2] = meshgrid(x1range, x2range);
    gridInputs = [X1(:), X2(:)];
    Z = reshape(predict(model, gridInputs), size(X1));

    predTest = predict(model, testInputs);
    rmse = sqrt(mean((testOutput - predTest).^2));

    fig = figure('Visible', 'off');

    surf(X1, X2, Z, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    hold on;
    colormap turbo;
    colorbar;

    scatter3(testInputs(:,1), testInputs(:,2), testOutput, ...
        8, testOutput, 'filled', 'MarkerFaceAlpha', 0.5);

    for i = 1:50:length(testInputs)
        plot3([testInputs(i,1) testInputs(i,1)], ...
              [testInputs(i,2) testInputs(i,2)], ...
              [predTest(i)     testOutput(i)], ...
              'r', 'LineWidth', 0.5);
    end

    xlabel(input1Label);
    ylabel(input2Label);
    zlabel(outputLabel);
    title([algoName ' — RMSE: ' num2str(round(rmse, 4))]);
    legend({'Model surface', 'Test data', 'Errors'}, 'Location', 'best');
    view(45, 30);
    grid on;

    fname = ['3D_' strrep(algoName, ' ', '_') '.png'];
    saveas(fig, fullfile(imageDir, fname));
    close(fig);
end