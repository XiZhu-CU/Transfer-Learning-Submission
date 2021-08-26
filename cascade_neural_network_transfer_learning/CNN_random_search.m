function bestModel = CNN_random_search(x_dat, y_dat, max_itr,cv_fold)

%% Random Searching for Hyperparameters Tunning in Brain Age Modeling
% Developed by C.L. Chen
%%  Hyperparameter settings:
% 1. # of hidden layers
% 2. # of neurons in hidden layers (2^n series as default setting)
% 3. regularization
% 4. activation function b/w hidden layers
 
%% Brain Age Modeling with Random Search (RS)

% max_itr = 300; % specify the max. round of RS
   
    key1 = 0; cp = 0;
    bestModel = [];
    bestMAE = Inf;
    dim_in = size(x_dat,2);

    while key1 == 0
        hp_numL = randperm( floor(log2(dim_in))-2,1)+2; % 1. # of hidden layers
        hp_numN = setneuro2(hp_numL,dim_in); % 2. # of neurons in hidden layers (2^n series as default setting)
        hp_regur = round(rand(1),2); % 3. regularization 
        hp_af = setaf2('Random'); % 4. activation function b/w hidden layers
    
    % K-fold CV
        array = CVfold(x_dat,y_dat,cv_fold);
        model_box = {};
        mae_box = [];
        for fold1 = 1:cv_fold
            temp_x_dat = array{1,fold1};
            temp_y_dat = array{3,fold1};
            temp_x_val = array{2,fold1};
            temp_y_val = array{4,fold1};
        
            numL = hp_numL;
            nNeuron = [hp_numN,2];
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
            CasNN.inputs{1}.size = size(temp_x_dat,2);
            for i=1:nLay
                CasNN.layers{i}.size = nNeuron(i);
            end
        % specify the activation function
            for i=1:nLay-1
                CasNN.layers{i}.transferFcn = hp_af;
            end
        % the activation function of the last layer, use 'purelin'
            CasNN.layers{nLay}.transferFcn = 'purelin';
        % min-max normalization
            CasNN.inputs{1,1}.processFcns{1,1} = 'mapminmax';
        % specify the optimizer
            CasNN.trainFcn = 'traingdx'; % 'trainscg' is faster
        % specify verbose logging or not
            CasNN.trainParam.showWindow = 0;
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
            CasNN.performParam.regularization = hp_regur;
        % specify the early stopping criterion
            CasNN.trainParam.max_fail = 100;
            
        % configure and train
            CasNN = configure(CasNN,temp_x_dat',temp_y_dat');
            Model1 = train(CasNN,temp_x_dat',temp_y_dat','useGPU','yes');
            y_hat_temp_val = Model1(temp_x_val')';
            mae_val = mean(abs(y_hat_temp_val-temp_y_val));
            model_box{1,fold1} = Model1;
            mae_box(1,fold1) = mae_val;
        end
    
        mae_avg = mean(mae_box);
        [~,loc] = min(abs(mae_box-mae_avg));
        model_sel = model_box{loc};
        if mae_avg < bestMAE
            bestModel = model_sel;
            fprintf('The MAE improved from %g to %g...\n',bestMAE,mae_avg);
            bestMAE = mae_avg;
        else
            fprintf('The MAE did not improve...\n');
        end
    
        cp = cp +1;
        if cp == max_itr
            key1 = 1;
        end
    end
    fprintf('Random Searching: Done!\n');
    fprintf('The "bestModel" is in the workspace\n');
end

 
 
