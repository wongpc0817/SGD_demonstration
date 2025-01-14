function [out] = newevaluation(targets, outputs)

nClasses=size(outputs,1);
beta=1;

avgAccuray=0.0;

errRate=0.0;

numeratorP=0.0;
denominatorP=0.0;

numeratorR=0.0;
denominatorR=0.0;

precisionMacro=0.0;

recallMacro=0.0;


[~,targetLabel]=max(targets,[],1);
[~,outputLabel]=max(outputs,[],1);
tp=zeros(nClasses);
tn=tp;
accuracy=0.0;
precision=0.0;
recall=0.0;

for i=1:nClasses
    %tp=0; tn=0;
    outputInd= (outputLabel==i);
    targetInd= (targetLabel==i);
    fp(i) = nnz(outputInd > targetInd);
    fn(i) = nnz(targetInd > outputInd);
    
    for j=1:size(outputs,2)
        if targetInd(j)==1 && outputInd(j)==1
            tp(i)=tp(i)+1;
        elseif targetInd(j)==0 && outputInd(j)==0
            tn(i)=tn(i)+1;
        end
        
    end    
    accuracy=accuracy+(tp(i)+tn(i))/size(outputs,2)/nClasses;
    if tp(i)+fp(i)~=0
    precision=precision+tp(i)/(tp(i)+fp(i))/nClasses;
    end
    if tp(i)+fn(i)~=0
    recall=recall+tp(i)/(tp(i)+fn(i))/nClasses;
    end
end

f1score=(2*precision*recall)/(precision+recall);

out.avgAccuracy = accuracy*100;
out.errRate = 100-out.avgAccuracy;
out.precisionMicro = precision*100;
out.recallMicro = recall*100;
out.fscoreMicro = f1score*100;
out.perf=[tp;tn;fp;fn];
out.precisionMacro = precision*100;
out.recallMacro = recall*100;
out.fscoreMacro = f1score*100;
end
