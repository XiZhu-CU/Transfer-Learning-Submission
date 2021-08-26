
function Model = CNN_manual(nNeuron,x_dat, y_dat)
% specify the number of layers and the neurons manually
% number of neurons in the hidden layers 
% or
% numL = 6; % number of layer
% nNeuron = round(linspace(size(x_dat,2),1,numL+1));
% nNeuron = nNeuron(2:end);
 
    nLay = length(nNeuron);
    CasNN = feedforwardnet;
    CasNN.numInputs = 1;
    CasNN.numLayers = nLay;
    CasNN.biasConnect = ones(1,nLay)';
    CasNN.inputConnect = ones(1,nLay)';
    CasNN.outputConnect = [zeros(1,nLay-1),1];
    CasNN.layerConnect = tril(ones(nLay),-1);
    CasNN.name = 'CasNN';
    for i = 1:nLay
        CasNN.layers{i}.name = ['L',num2str(i)];
    end
    CasNN.inputs{1}.size = size(x_dat,2);
    for i=1:nLay
        CasNN.layers{i}.size = nNeuron(i);
    end
 
% specify the activation function
    for i=1:nLay-1
        CasNN.layers{i}.transferFcn = 'logsig'; 
    end
    CasNN.layers{nLay}.transferFcn = 'purelin';
 
% min-max normalization
    CasNN.inputs{1,1}.processFcns{1,1} = 'mapminmax'; 
% specify the optimizer
    CasNN.trainFcn = 'traingdx';
% specify verbose logging or not
    CasNN.trainParam.showWindow = 1;
% specify the loss function
    CasNN.performFcn = 'mse';
% specify error normalization
    CasNN.performParam.normalization = 'standard';
% specify number of epoch
    CasNN.trainParam.epochs = 1000; 
    CasNN.plotFcns = {'plotperform'};
% specify the proportion of hold-out validatoin for early stopping
    CasNN.divideFcn = 'dividerand'; 
    CasNN.divideParam.testRatio = 0;
    CasNN.divideParam.trainRatio = 0.85;
    CasNN.divideParam.valRatio = 0.15;
% specify the degree of regularization
    CasNN.performParam.regularization = 0.65;
% specify the early stopping criterion
    CasNN.trainParam.max_fail = 100;
 
% configure and train
    CasNN = configure(CasNN,x_dat',y_dat');
%[Model1, train_rec] = train(CasNN,x_dat',y_dat','useGPU','yes');
    [Model, train_rec] = train(CasNN,x_dat',y_dat','useGPU','yes');
    fprintf('Done!\n')
end
 


