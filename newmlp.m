function output = newmlp(app,features, config)
resultFile='training/results.csv';

featuresInput=features.Input;
featuresTarget=features.Target;


 p = randperm(size(featuresInput,1));
 trainR = 0.7;
 testR =  0.3;
 valR =   0;
 featuresInput = featuresInput(p,:);
 featuresTarget= featuresTarget(p,:);
 [trainInd,valInd,testInd] = dividerand(size(featuresInput,1),trainR,valR,testR);
 
 features.trainInd=trainInd;
 features.valInd=valInd;
 features.testInd=testInd;

% Normalise the Dataset
if app.normalise
    for i = 1:size(featuresInput,1)
        featuresInput(i,:) = (featuresInput(i,:)-mean(featuresInput(i,:)))./std(featuresInput(i,:));
    end
end

% Set up epochs, goal, layers ... 
config.inputs = featuresInput(trainInd,:);
config.targets = featuresTarget(trainInd,:);
config.goal = 1e-10;

%SGD,LADA,Adam,HBMom,AbsAdam,ExpAdam,L1_4

% Train the Networks
disp('training MLP...')
tstart=tic;
net = trainMLP(app,config);
telapsed=toc(tstart);

% Test the Network
disp('testing newMLP...');
testInputs=featuresInput(testInd,:);
testTargets=featuresTarget(testInd,:);
outputs = applynetwork(net,testInputs);

disp('finally evaluating')
[~,cm,~,~] = confusion(testTargets', outputs);
results = newevaluation(testTargets', outputs);
results.time = num2str(telapsed);
results.epoches = num2str(net.epochs(end));
results.hLayerNeurons = config.layers;
results.datasetName = '';
results.numTrainSamples = num2str(size(config.inputs,1));
results.numValSamples = num2str(0);
results.numTestSamples = num2str(size(testInputs,1));
results.algo = 'MLP';
results.gradAlgo = config.gradAlgo;
results.errorRecord=net.error;
results.cm=cm;
output.results=results;
output.net=net;
end
