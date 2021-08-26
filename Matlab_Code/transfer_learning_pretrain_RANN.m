addpath /Volume/PTSD/toolbox/cascade_neural_network_transfer_learning;

load('rann_fs7_withdemo.mat')


%% RANN Data Pretraining

% Split data into 70% Train, 30% Test.
test_prop=0.3;
[dataTest, dataTrain]=splitdata(rann_fs7,test_prop, 202428);

behavior_train=dataTrain(:,1:11);
behavior_test=dataTest(:,1:11);

dataTrain(:,[1,3,4,5,6,7,8,11])=[];
dataTest(:,[1,3,4,5,6,7,8,11])=[];
dataTrain=table2cell(dataTrain);
dataTest=table2cell(dataTest);

% Specify the features and responses for the training and test sets 
x_train = cell2mat(dataTrain(:,2:end)); % the input feature for training set
y_train = cell2mat(dataTrain(:,1)); % the response for training set
x_test = cell2mat(dataTest(:,2:end)); % the input feature for test set
y_test = cell2mat(dataTest(:,1)); % the response for test set
%included age and sex, not include TIV

nNeuron = [64,32,16,8,4,2,1];

%[x_train_subset,y_train_subset] = random_select(x_train,y_train,346);
model_rann_manual=CNN_manual(nNeuron, x_train, y_train);

view(model_rann_manual)
% CasNN model inference
y_hat_train_rann = model_rann_manual(x_train')';
model_inference(y_hat_train_rann,y_train,1);
 
y_hat_test_rann = model_rann_manual(x_test')';
model_inference(y_hat_test_rann,y_test,2);
 

%% Random Search Model
rng(2);
bestModel_rann_rs=CNN_random_search(x_train, y_train, 100, 10);

view(bestModel_rann_rs)
y_hat_train_rs_rann = bestModel_rann_rs(x_train')';
model_inference(y_hat_train_rs_rann,y_train,1);
 
y_hat_test_rs_rann = bestModel_rann_rs(x_test')';
model_inference(y_hat_test_rs_rann,y_test,2);

RANN_train_residual=y_hat_train_rs_rann-y_train;
RANN_test_residual=y_hat_test_rs_rann-y_test;


%check correlation between residual and memory, education and IQ
RANN_train_residual(:,2)=cell2mat(behavior_train.npmemory);
RANN_train_residual(:,3)=behavior_train.Education;
RANN_train_residual(:,4)=behavior_train.NARTIQ;

name=["Res";"Mem"; "Edu"; "IQ"];
RANN_train_residual2= array2table(RANN_train_residual);
RANN_train_residual2.Properties.VariableNames(1:4)=name;
[R,P]=corrplot(RANN_train_residual2,'type','Pearson','testR','on')

RANN_test_residual(:,2)=cell2mat(behavior_test.npmemory);
RANN_test_residual(:,3)=behavior_test.Education;
RANN_test_residual(:,4)=behavior_test.NARTIQ;
RANN_test_residual2= array2table(RANN_test_residual);
RANN_test_residual2.Properties.VariableNames(1:4)=name;
[R,P]=corrplot(RANN_test_residual2,'type','Pearson','testR','on')

save('pretrained_mem_model_rann.mat','model_rann_manual','bestModel_rann_rs')


%% Transfer Learning to ADNI /HCP Data

load('adni_fs7_withdemo.mat'); 
data=ADNIall;

% load('hcpa_fs7_n620_withdemo.mat'); 
% data=hcpafs7n620withdemo;

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


% Directly apply tre-trained model without tuning
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

%correlation with IQ and EDU, paper data and thing

edu=table2cell(behavior_train(:,4));
IQ=table2cell(behavior_train(:,5));
edu=fillmissing(edu,'linear');
IQ=fillmissing(IQ,'linear');
edu=cell2mat(edu);
IQ=cell2mat(IQ);


[R_edu,P_edu]=corrcoef(ADNI_train_residual,edu);
[R_IQ,P_IQ]=corrcoef(ADNI_train_residual,IQ);

edu=table2cell(behavior_test(:,4));
IQ=table2cell(behavior_test(:,5));

edu=cell2mat(edu);
IQ=cell2mat(IQ);
edu=fillmissing(edu,'linear');
IQ=fillmissing(IQ,'linear');

[R_edu,P_edu]=corrcoef(ADNI_test_residual,edu);
[R_IQ,P_IQ]=corrcoef(ADNI_test_residual,IQ);


[R,P]=corrcoef(A(:,1),A(:,2))


%% Transfer Learning to HCP Aging Data

% load('hcpa_fs7_n620.mat'); 
% [hcpa_fs7_n620_withdemo,ileft,iright] = outerjoin(hcpa_fs7_n620,hcpamergeddemocogv120210417,'Type','left','Keys','SubID');

%hcpa_fs7_n620 = removevars(hcpa_fs7_n620, 'SubID');
%hcpa=table2cell(hcpafs7n620withdemo);

% %% Manual model Transfer
% pt_model = model_rann_manual; % specify the pre-trained model
% %pt_model =bestModel_rann_rs;
% %%%   pre-trained model settings  %%%
% % Optimizer
% % Optimizer
% pt_model.trainFcn = 'traingdx'; % 'trainscg' is faster
% pt_model.trainParam.epochs = 1000; 
% pt_model.trainParam.max_fail = 100;
% pt_model.trainParam.showWindow = 1;
% pt_model.performFcn = 'mse'; % mae, mse, sae, sse, etc.
% pt_model.performParam.regularization = 0.1; % 0.25, 0.1, 0.01, ect.
% z=zeros(1,pt_model.numLayers);
% fot = logical(z); 
% pt_model = freezelayer(pt_model,fot);
% Model_TL = train(pt_model,x_hcpatune',y_hcpatune','useGPU','yes'); % training
% 
% % Model inference
% y_hat_hcpatune = Model_TL(x_hcpatune')';
% model_inference(y_hat_hcpatune,y_hcpatune,1);
%  
% y_hat_hcpatest = Model_TL(x_hcpatest')';
% model_inference(y_hat_hcpatest,y_hcpatest,2);
% 
% HCPA_tune_residual=y_hat_hcpatune-y_hcpatune;
% HCPA_test_residual=y_hat_hcpatest-y_hcpatest;
% 
% % Directly apply tre-trained model without tuning
% y_hat_hcpatune2 = model_rann_manual(x_hcpatune')';
% model_inference(y_hat_hcpatune2,y_hcpatune,3);
% 
% y_hat_hcpatest2 = model_rann_manual(x_hcpatest')';
% model_inference(y_hat_hcpatest2,y_hcpatest,3);


edu=behavior_hcpa_train.moca_edu;
IQ1=behavior_hcpa_train.nih_fluidcogcomp_unadjusted;
IQ2=behavior_hcpa_train.nih_crycogcomp_unadjusted;

edu=behavior_hcpa_test.moca_edu;
IQ1=behavior_hcpa_test.nih_fluidcogcomp_ageadjusted;
IQ2=behavior_hcpa_test.nih_crycogcomp_ageadjusted;

edu=behavior_hcpa_test.moca_edu;
IQ1=behavior_hcpa_test.nih_fluidcogcomp_unadjusted;
IQ2=behavior_hcpa_test.nih_crycogcomp_unadjusted;


edu=fillmissing(edu,'linear');
IQ1=fillmissing(IQ1,'linear');
IQ2=fillmissing(IQ2,'linear');


[R_edu,P_edu]=corrcoef(HCPA_tune_residual,edu);
[R_IQ1,P_IQ1]=corrcoef(HCPA_tune_residual,IQ1);
[R_IQ2,P_IQ2]=corrcoef(HCPA_tune_residual,IQ2);


[R_edu,P_edu]=corrcoef(HCPA_test_residual,edu);
[R_IQ1,P_IQ1]=corrcoef(HCPA_test_residual,IQ1);
[R_IQ2,P_IQ2]=corrcoef(HCPA_test_residual,IQ2);


edu=table2cell(behavior_test(:,4));
IQ=table2cell(behavior_test(:,5));

edu=cell2mat(edu);
IQ=cell2mat(IQ);
edu=fillmissing(edu,'linear');
IQ=fillmissing(IQ,'linear');

[R_edu,P_edu]=corrcoef(ADNI_test_residual,edu);
[R_IQ,P_IQ]=corrcoef(ADNI_test_residual,IQ);
