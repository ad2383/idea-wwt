%Load a Data Set
load energyDataset.mat

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