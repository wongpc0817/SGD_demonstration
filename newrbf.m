function output = newrbf(app,features, config)
resultFile='training/results.csv';

featuresInput=features.Input;
featuresTarget=features.Target;

% Normalise the Dataset
if app.normalise
    for i = 1:size(featuresInput,1)
        featuresInput(i,:) = (featuresInput(i,:)-mean(featuresInput(i,:)))./std(featuresInput(i,:));
    end
end

p = randperm(size(featuresInput,1));
trainR = 0.7;
testR =  0.3;
valR =   0;
featuresInput = featuresInput(p,:);
featuresTarget= featuresTarget(p,:);
[trainInd,valInd,testInd] = dividerand(size(featuresInput,1),trainR,valR,testR);


% Set up epochs, goal, layers ... 
config.inputs = featuresInput(trainInd,:);
config.targets = featuresTarget(trainInd,:);
config.goal = 1e-10;
%SGD,LADA,Adam,HBMom,ExpAdam,AbsAdam,L1_4

% Train the Networks
    disp('training newRBF...')
    tstart=tic;
    net = trainRBF(app,config);
    telapsed=toc(tstart);

% Test the Network
disp('testing newRBF...');
testInputs = featuresInput(testInd,:);
testTargets = featuresTarget(testInd,:);
outputs = applynetwork(net,testInputs);

disp('finally evaluating')
results = newevaluation(testTargets', outputs);
[~,cm,~,~] = confusion(testTargets', outputs);
results.cm=cm;
results.time =num2str(telapsed);
results.epoches = num2str(net.epochs);
results.hLayerNeurons = config.layers;
results.datasetName = '';
results.numTrainSamples = num2str(size(config.inputs,1));
results.numValSamples = num2str(0);
results.numTestSamples = num2str(size(testInputs,1));
results.algo = 'RBF';
results.gradAlgo = config.gradAlgo;
results.errorRecord=net.error;

output.results=results;
output.net=net;
end
