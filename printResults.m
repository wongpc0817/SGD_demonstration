function printResults(app,table,results)
app.Value{1}=strcat('Model   : ',results.algo);
app.Value{2}=strcat('Learning Algorithm   :',results.gradAlgo);
app.Value{3}=strcat('Number of Training Sample   : ',results.numTrainSamples);
app.Value{4}=strcat('Number of Testing Sample   : ',results.numTestSamples);
app.Value{5}=strcat('Network Training Time(sec)    : ', results.time);
app.Value{6}=strcat('Average System Accuracy(%)   : ', num2str(results.avgAccuracy));
app.Value{7}=strcat('Precision (%)         : ', num2str(results.precisionMicro));
app.Value{8}=strcat('Recall (%)            : ', num2str(results.recallMicro));
app.Value{9}=strcat('Fscore (%)            : ', num2str(results.fscoreMicro));
CM=results.cm;
table.Data=array2table(CM);
table.ColumnName=num2cell(1:size(CM,1));
table.ColumnWidth=num2cell(ones(1,size(CM,1))*40);

end