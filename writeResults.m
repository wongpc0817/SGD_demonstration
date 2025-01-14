function writeResults(results, fileName)
trainSize = results.numTrainSamples;
testSize = results.numTestSamples;
Model=results.algo;
recognitionAlgo = results.gradAlgo;
epoches = results.epoches;
hLayerNeurons = num2str(results.hLayerNeurons);
tTime = num2str(results.time);
avgAccuray = sprintf('%.2f', results.avgAccuracy);
precisionMicro = sprintf('%.2f', results.precisionMicro);
recallMicro = sprintf('%.2f', results.recallMicro);
fscoreMicro = sprintf('%.2f', results.fscoreMicro);

sep = ',';
if exist(fileName, 'file') == 0
    header = strcat( 'Training Samples', sep,  ...
      'Testing Samples', sep,'Model',sep, 'Recognition Algorithm', sep, 'Hidden Layer Neurons', sep, ...
      'No. of Epoches', sep, 'Training Time(sec)', sep, 'Avg. System Accuracy(%)', sep,  ...
      'Precesion(%)', sep, 'Recall(%)', sep, 'F1score(%)');
 
    data = strcat( trainSize, sep, ...
      testSize, sep, Model,sep,recognitionAlgo, sep, hLayerNeurons, sep, ...
      epoches, sep, tTime, sep, avgAccuray, sep,  ...
      precisionMicro, sep, recallMicro, sep, fscoreMicro);
 
    fid = fopen(fileName, 'w');
    fprintf(fid, '%s\n', header);
    fprintf(fid, '%s\n', data);
    fclose(fid);
else
    data = strcat(trainSize, sep, ...
      testSize, sep, Model,sep ,recognitionAlgo, sep, hLayerNeurons, sep, ...
      epoches, sep, tTime, sep, avgAccuray, sep,  ...
      precisionMicro, sep, recallMicro, sep, fscoreMicro);
 
    fid = fopen(fileName, 'a');
    fprintf(fid, '%s\n', data);
    fclose(fid);
end
end
