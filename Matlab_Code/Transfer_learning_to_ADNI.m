%% Transfer Learning to ADNI /HCP Data

load('adni_fs7_withdemo.mat'); 
data=ADNIall;

% Split data into 70% Train, 30% Test.
view(bestModel_rann_rs)

test_prop=0.3;
[TargetTest, TargetTune]=splitdata(data,test_prop, 20210430);

%ADNI data:
behavior_train=TargetTune(:,1:8);
behavior_test=TargetTest(:,1:8);

TargetTune(:,[1,2,4,5,6])=[];
TargetTest(:,[1,2,4,5,6])=[];

TargetTune=table2cell(TargetTune);
TargetTest=table2cell(TargetTest);

x_Targettune = cell2mat(TargetTune(:,2:end)); % the input feature for training set
y_Targettune = cell2mat(TargetTune(:,1)); % the response for training set
x_Targettest = cell2mat(TargetTest(:,2:end)); % the input feature for test set
y_Targettest = cell2mat(TargetTest(:,1)); % the response for test set

x_Targettune(:,end) = 2; % change the site indicator (default: CamCAN: 1)
x_Targettest(:,end) = 2; % change the site indicator (default: CamCAN: 1)

%% Using the random subset to fine-tune the model

%[x_Targettune,y_Targettune] = random_select(x_Targettune,y_Targettune,400);

%% Manual model Transfer
pt_model = bestModel_rann_rs; % specify the pre-trained model
 
%%%   pre-trained model settings  %%%
% Optimizer
pt_model.trainFcn = 'traingdx'; % 'trainscg' is faster
pt_model.trainParam.epochs = 500; 
pt_model.trainParam.max_fail = 75;
pt_model.trainParam.showWindow = 1;
pt_model.performFcn = 'mse'; % mae, mse, sae, sse, etc.
pt_model.performParam.regularization = 0.01; % 0.25, 0.1, 0.01, ect.
z=zeros(1,pt_model.numLayers);
fot = logical(z); 
pt_model = freezelayer(pt_model,fot);
Model_TL = train(pt_model,x_Targettune',y_Targettune','useGPU','yes'); % training
% Model inference
y_hat_Targettune = Model_TL(x_Targettune')';
model_inference(y_hat_Targettune,y_Targettune,1);
 
y_hat_Targettest = Model_TL(x_Targettest')';
model_inference(y_hat_Targettest,y_Targettest,2);

train_residual=y_hat_Targettune-y_Targettune;
test_residual=y_hat_Targettest-y_Targettest;


% Directly apply the pre-trained model without tuning
y_hat_Targettune2 = model_rann_manual(x_Targettune')';
model_inference(y_hat_Targettune2,y_Targettune,3);

y_hat_Targettest2 = model_rann_manual(x_Targettest')';
model_inference(y_hat_Targettest2,y_Targettest,3);


%retrain the model using source data and target data

%% TLCO Integrate those two datasets from both source and target domains
x_tune = [x_train;x_Targettune]; %source and target
y_tune = [y_train;y_Targettune]; 

%% Transfer Learning Approach
%%%   pre-trained model settings  %%%
% Optimizer
pt_model.trainFcn = 'traingdx'; % 'trainscg' is faster
% Epochs
pt_model.trainParam.epochs = 700; 
% Tolerance of early stopping
pt_model.trainParam.max_fail = 500; % to force the model going through at least 500 epos
% Verbose
pt_model.trainParam.showWindow = 1;
% Loss function
pt_model.performFcn = 'mse'; % mae, mse, sae, sse, etc.
% Regularization
pt_model.performParam.regularization = 0.01; % 0.25, 0.1, 0.01, ect.
% Freeze partial layers or not (demo)
z=zeros(1,pt_model.numLayers);
fot = logical(z); 
%fot = logical([0,0,0,0,0,0,0,0]); % specify which layers should be frozen (frozen: 1).
% make sure the length of the above vector have the same size with the
% number of layers in the model (use Model.numLayers to check)
% [0,0,0,1,1,1] means the fourth to the last layers (close to the input layer) are frozen.
pt_model = freezelayer(pt_model,fot);

Model_TL = train(pt_model,x_tune',y_tune','useGPU','yes'); % training
view(Model_TL)

%% Model inference
% Prediction on two sites at once
% Model inference

% Target domain
y_hat_tune = Model_TL(x_Targettune')';
model_inference(y_hat_tune,y_Targettune,1);
y_hat_test = Model_TL(x_Targettest')';
model_inference(y_hat_test,y_Targettest,2);

% Source domain
y_hat_tune = Model_TL(x_train')';
model_inference(y_hat_tune,y_train,3);
y_hat_test = Model_TL(x_test')';
model_inference(y_hat_test,y_test,4);

tune_residual=y_hat_Targettune-y_Targettune;
test_residual=y_hat_Targettest-y_Targettest;

