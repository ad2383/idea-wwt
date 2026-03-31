%Load a Data Set
scriptDir = fileparts(mfilename('fullpath'));
load(fullfile(scriptDir, '..', 'data', 'energyDataset.mat'))

%We will use every possible variable to create a multi-linear regression
%model of Energy Consumption
TrainingMatrix = [AppliancesUsed, AverageTemperature, NumberOfOccupants, SquareFootage];

%Now when we train our algorithm
linearModel = fitlm(TrainingMatrix,EnergyConsumption);

%Applies the model to our x-variable to produce the model's result
EnergyModel = predict(linearModel,TrainingMatrix);

%Calculates Mean Square Error
MSE = 0;
N = length(EnergyConsumption);
for i = 1:1:N
    MSE = MSE + (EnergyConsumption(i)-EnergyModel(i))^2;
end

%Compare this MSE to the single variable linear regression
MSE = MSE/N;

figure;
plot(EnergyConsumption, 'o', 'MarkerSize', 6, 'LineWidth', 1.5);
hold on;
plot(EnergyModel, 'x', 'MarkerSize', 6, 'LineWidth', 1.5);
xlabel('Sample Index');
ylabel('Energy Consumption (kWh)');
title(['Multi-Linear Regression - MSE: ' num2str(round(100*MSE)/100) 'kWh^2']);
legend('Actual', 'Predicted', 'Location', 'best');
grid on;

saveas(gcf, fullfile(scriptDir, '..', 'images', 'linearRegression_multi.png'));