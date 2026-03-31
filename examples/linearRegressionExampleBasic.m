%Load a Data Set
scriptDir = fileparts(mfilename('fullpath'));
load(fullfile(scriptDir, '..', 'data', 'energyDataset.mat'))
x= SquareFootage; %We will use square footage to train this model.
%Feel free to change this variable to another one found in the workspace
%when you load energyDataset.mat

%Now when we train our algorithm
linearModel = fitlm(x,EnergyConsumption);

%Applies the model to our x-variable to produce the model's result
EnergyModel = predict(linearModel,x);

%Calculates Mean Square Error
MSE = 0;
N = length(EnergyConsumption);
for i = 1:1:N
    MSE = MSE + (EnergyConsumption(i)-EnergyModel(i))^2;
end

MSE = MSE/N;

%Plots the model vs training data with error margins
plot(x,EnergyConsumption,'LineStyle','none','Marker','o')
hold on
plot(x,EnergyModel,'Color',[1 0 1])
for i = 1:1:N
    plot([x(i) x(i)],...
        [EnergyModel(i),EnergyConsumption(i)],'Color',[1 0 0])
end
xlabel('Square Footage (ft^2)')
ylabel('Energy Consumption (kWh')

title(['Linear Regression Model - Square Footage predicting' ...
    ' Energy Consumption - MSE: ' num2str(round(100*MSE)/100) 'kWh^2'])

legend('Original Data','Model','Error Margins','Location','northwest')

saveas(gcf, fullfile(scriptDir, '..', 'images', 'linearRegression_basic.png'));