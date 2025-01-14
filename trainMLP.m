function network = trainMLP(app,config)
% Format: config.inputs = SxQ matrix
%         config.targets = SxU matrix
%         config.goal = stopping condition (relative error)
%         config.epochs = iterations
%         config.layers = 1xN matrix, each element refers to the
%         number of hidden neurons on that layer.
%         config.gradAlgo = Gradient Search Algorithm Used Here
%         network.weights = MxMxN matrix
%         network.model = 'MLP'
%         network.epochs = epochs needed
%         network.layers = config.layers 1xN matrix
%         network.gradAlgo = Gradient Search Algorithms Used Here
%         network.struct = layers [Ni,Ci];
%         network.bias = MxN matrix
%S = number of datasets,
%Q = number of inputs per dataset
%U = number of outputs
%M = maximum number of neurons
%N = number of layers

network.model = 'MLP';
noflayers = size(config.layers,2)+2; %including input and output layers
goal = config.goal;
network.layers = config.layers;
epochs = config.epochs;
x = config.inputs;
network.trainsize = size(x,1);
tar = config.targets;
algo = config.gradAlgo;
% Step 1 - Feed Forward Phase
noutput = size(tar,2);

format long;
maximum= max(config.layers);
maximum = max([maximum,size(x,2),noutput]);
weights=2*rand(maximum,maximum,noflayers-1)-1;% weight matrix
weights(:,:,noflayers-1)=weights(:,:,noflayers-1)./2;
bias=2*ones(maximum,1)*rand(1,noflayers-1)-1;
layer=zeros(noflayers-1,2);
ninput=size(x,1);
p = randperm(ninput);
x = x(p,:);
elesize=size(x,2);
tar = tar(p,:);
if size(tar,1)~=size(x,1)
    disp('Number of samples do not match.');
end
% Set parameters
alpha = config.alpha;
momentum = 0.9;
beta1 = 0.9;
beta2 = 0.999;
momentump = 0.8;
epsilon = 1e-8;
reg = 10;
for i=1:noflayers-1
    if i==1
        layer(i,2)=size(x,2);
        layer(i,1)=config.layers(i);
    elseif i==noflayers-1
        layer(i,2)=config.layers(i-1);
        layer(i,1)=noutput;
    else
        layer(i,2)=config.layers(i-1);
        layer(i,1)=config.layers(i);
    end
end
output=zeros(ninput,noutput);
network.struct = layer;
phi=zeros(layer(end,2)+1,ninput);
errorRecord = 0;

%for attempt=1:5
% weights=zeros(maximum,maximum,noflayers-1)+0.5;% weight matrix
weights(:,:,noflayers-1)=weights(:,:,noflayers-1)./2;
bias=2*ones(maximum,1)*rand(1,noflayers-1)-1;
% bias=zeros(maximum,noflayers-1)+0.5;
success = 0;
e=0;
op=zeros(maximum,noflayers-1);
ninput=size(x,1);
bias=2*rand(maximum,noflayers-1)-1;
delw=zeros(size(weights));
delb=zeros(maximum,noflayers-1);
pdelw=zeros(size(delw));
pdelb=zeros(size(delb));
mt=zeros(maximum,maximum,noflayers-1);
vt=zeros(maximum,maximum,noflayers-1);
mt1=zeros(maximum,noflayers-1);
vt1=zeros(maximum,noflayers-1);
normConst = 3;
lambda = 1e-6;
Pt=mt;
Pmt=mt+epsilon;
Pt1=mt1;
Pmt1=mt1+epsilon;
for epoch=1:epochs
    success=0;
    e=0;
    for i_x=1:ninput
        pdelw=delw;
        pdelb=delb;
        delw=0*delw;
        delb=0*delb;
        intemp=x(i_x,:);
        for k=1:noflayers-1
            N=layer(k,1);C=layer(k,2);
            opin(1:N,k)=bias(1:N,k)+weights(1:N,1:C,k)*intemp';
            op(1:N,k)=sigmoid(opin(1:N,k));
            intemp=op(1:N,k)';
        end
        %error
        outtemp(i_x,:)=intemp;
        e=e+sum((tar(i_x,:)-intemp).^2)/2/ninput;
        [~,I]=max(intemp);
        [~,label]=max(tar(i_x,:));
        if I==label
            success=success+1;
        end
        
        %Backpropagation
        for k=noflayers-1:-1:1
            N=layer(k,1);C=layer(k,2);
            if k==noflayers-1
                del(1:N,k)=(tar(i_x,:)'-op(1:N,k)).*sigmoid1(op(1:N,k));
            else
%                 del(1:N,k)=dot(del(1:n,k+1),weights(1:n,1:N,k+1)).*sigmoid1(op(1:N,k));
                for i=1:N
                    del(i,k)=dot(del(1:n,k+1),weights(1:n,i,k+1))*sigmoid1(op(i,k));
                end
            end
            n=N;c=C;
        end
        switch algo
            case 'SGD'
 %               alpha=0.01;
                momentum=0.9;
                for k=1:noflayers-1
                    N=layer(k,1);C=layer(k,2);
                    if k==1
                        temp=x(i_x,:);
                    else
                        temp=op(1:C,k-1)';
                    end
                    delw(1:N,1:C,k)=alpha*del(1:N,k)*temp;
                    delb(1:N,k)=alpha*del(1:N,k);
                    n=N;c=C;
                end
                weights=weights+delw+momentum*pdelw;
                bias=bias+delb+momentum*pdelb;
            case 'Adam'
%                 beta1=0.9;
%                alpha=1e-6;
                momentum=0.8;
                for k=1:noflayers-1
                    N=layer(k,1);C=layer(k,2);
                    if k==1
                        temp=x(i_x,:);
                    else
                        temp=op(1:C,k-1)';
                    end
                    grad=del(1:N,k)*temp;
                    if epoch==1 && i_x==1
                        mt(1:N,1:C,k)=grad;
                        vt(1:N,1:C,k)=grad.^2;
                    else
                        mt(1:N,1:C,k)=beta1*mt(1:N,1:C,k)+(1-beta1)*grad;
                        vt(1:N,1:C,k)=beta2*vt(1:N,1:C,k)+(1-beta2)*grad.^2;
                    end
                    if epoch==1 && i_x==1
                        mt1(1:N,k)=del(1:N,k); vt1(1:N,k)=del(1:N,k).^2;
                    else
                        mt1(1:N,k)=beta1*mt1(1:N,k)+(1-beta1)*del(1:N,k);
                        vt1(1:N,k)=beta2*vt1(1:N,k)+(1-beta2)*del(1:N,k).^2;
                    end
                end
                mth=mt./(1-beta1^epoch);
                vth=vt./(1-beta2^epoch);
                delw=alpha*mth./(sqrt(vth)+epsilon);
                
                mth1=mt1./(1-beta1^epoch);
                vth1=vt1./(1-beta2^epoch);
                delb=alpha*mth1./(sqrt(vth1)+epsilon);
                
                weights=weights+delw;
                bias=bias+delb;
            case 'HBMom'
  %              alpha=3e-5;
                momentum=0.0;
                for k=1:noflayers-1
                    N=layer(k,1);C=layer(k,2);
                    if k==1
                        temp=x(i_x,:);
                    else
                        temp=op(1:C,k-1)';
                    end
                    grad=del(1:N,k)*temp;
                    if epoch~=1 || i_x~=1
                        mt(1:N,1:C,k)=((mt(1:N,1:C,k).*grad)>=0);
                        mt1(1:N,k)=(mt1(1:N,k).*del(1:N,k)>=0);
                    end
                    delw(1:N,1:C,k)=alpha*grad;
                    delb(1:N,k)=alpha*del(1:N,k);
                    n=N;c=C;
                end
                weights=weights+delw+momentum*pdelw.*mt;
                bias=bias+delb+momentum*pdelb.*mt1;
            case 'AbsAdam'
   %             alpha=2.5e-5;
                for k=1:noflayers-1
                    N=layer(k,1);C=layer(k,2);
                    if k==1
                        temp=x(i_x,:);
                    else
                        temp=op(1:C,k-1)';
                    end
                    grad=del(1:N,k)*temp;
                    if epoch==1 && i_x==1
                        mt(1:N,1:C,k)=grad;
                        vt(1:N,1:C,k)=abs(grad);
                    else
                        mt(1:N,1:C,k)=beta1*mt(1:N,1:C,k)+(1-beta1)*grad;
                        vt(1:N,1:C,k)=beta2*vt(1:N,1:C,k)+(1-beta2)*abs(grad);
                    end
                    if epoch==1 && i_x==1
                        mt1(1:N,k)=del(1:N,k); vt1(1:N,k)=abs(del(1:N,k));
                    else
                        mt1(1:N,k)=beta1*mt1(1:N,k)+(1-beta1)*del(1:N,k);
                        vt1(1:N,k)=beta2*vt1(1:N,k)+(1-beta2)*abs(del(1:N,k));
                    end
                end
                mth=mt./(1-beta1^epoch);
                vth=vt./(1-beta2^epoch);
                delw=alpha*mth./(sqrt(vth)+epsilon);
                mth1=mt1./(1-beta1^epoch);
                vth1=vt1./(1-beta2^epoch);
                delb=alpha*mth1./(sqrt(vth1)+epsilon);
                weights=weights+delw;
                bias=bias+delb;
                
            case 'ExpAdam'
    %            alpha=2.5e-5;
                for k=1:noflayers-1
                    N=layer(k,1);C=layer(k,2);
                    if k==1
                        temp=x(i_x,:);
                    else
                        temp=op(1:C,k-1)';
                    end
                    grad=del(1:N,k)*temp;
                    if epoch==1 && i_x==1
                        mt(1:N,1:C,k)=grad;
                        vt(1:N,1:C,k)=exp(abs(grad));
                    else
                        mt(1:N,1:C,k)=beta1*mt(1:N,1:C,k)+(1-beta1)*grad;
                        vt(1:N,1:C,k)=beta2*vt(1:N,1:C,k)+(1-beta2)*exp(abs(grad));
                    end
                    if epoch==1 && i_x==1
                        mt1(1:N,k)=del(1:N,k); vt1(1:N,k)=exp(abs(del(1:N,k)));
                    else
                        mt1(1:N,k)=beta1*mt1(1:N,k)+(1-beta1)*del(1:N,k);
                        vt1(1:N,k)=beta2*vt1(1:N,k)+(1-beta2)*exp(abs(del(1:N,k)));
                    end
                end
                mth=mt./(1-beta1^epoch);
                vth=vt./(1-beta2^epoch);
                delw=alpha*mth./log(vth+epsilon);
                mth1=mt1./(1-beta1^epoch);
                vth1=vt1./(1-beta2^epoch);
                delb=alpha*mth1./log(vth1+epsilon);
                weights=weights+delw;
                bias=bias+delb;
            case 'LADA'
     %           alpha=2.5e-5;
                momentump=0.9;
                normC=zeros(maximum,maximum,noflayers-1);
                normC1=zeros(maximum,noflayers-1);
                for k=1:noflayers-1
                    N=layer(k,1);C=layer(k,2);
                    if k==1
                        temp=x(i_x,:);
                    else
                        temp=op(1:C,k-1)';
                    end
                    grad=del(1:N,k)*temp;
                    if epoch==1 && i_x==1
                        mt(1:N,1:C,k)=grad;
                        vt(1:N,1:C,k)=grad.^2;
                        Pt(1:N,1:C,k)=abs(grad);
                    else
                        Pmt(1:N,1:C,k)=max(Pmt(1:N,1:C,k),abs(grad));
                        Pt(1:N,1:C,k)=momentump*Pt(1:N,1:C,k)+(1-momentump)*abs(grad);
                        normC(1:N,1:C,k)=3-Pt(1:N,1:C,k)./Pmt(1:N,1:C,k);
                        mt(1:N,1:C,k)=beta1*mt(1:N,1:C,k)+(1-beta1)*grad;
                        vt(1:N,1:C,k)=beta2*vt(1:N,1:C,k)+(1-beta2)*abs(grad).^normC(1:N,1:C,k);
                    end
                    if epoch==1
                        mt1(1:N,k)=del(1:N,k);
                        vt1(1:N,k)=del(1:N,k).^2;
                        Pt1(1:N,k)=abs(del(1:N,k));
                        
                    else
                        Pmt1(1:N,k)=max(Pmt1(1:N,k),abs(del(1:N,k)));
                        Pt1(1:N,k)=momentump*Pt1(1:N,k)+(1-momentump)*abs(del(1:N,k));
                        normC1(1:N,k)=3-Pt1(1:N,k)./Pmt1(1:N,k);
                        mt1(1:N,k)=beta1*mt1(1:N,k)+(1-beta1)*del(1:N,k);
                        vt1(1:N,k)=beta2*vt1(1:N,k)+(1-beta2)*abs(del(1:N,k)).^normC1(1:N,k);
                    end
                end
                mth=mt./(1-beta1^epoch);
                vth=vt./(1-beta2^epoch);
                delw=alpha*mth./(vth.^(1./normC)+epsilon);
                mth1=mt1./(1-beta1^epoch);
                vth1=vt1./(1-beta2^epoch);
                delb=alpha*mth1./(vth1.^(1./normC1)+epsilon);
                weights=weights+delw;
                bias=bias+delb;
            case 'L1_4'
      %          alpha=2.5e-5;
                lambda=1e-3;
                normC=zeros(maximum,maximum,noflayers-1);
                normC1=zeros(maximum,noflayers-1);
                for k=1:noflayers-1
                    N=layer(k,1);C=layer(k,2);
                    if k==1
                        temp=x(i_x,:);
                    else
                        temp=op(1:C,k-1)';
                    end
                    grad=del(1:N,k)*temp;
                    if epoch==1 && i_x==1
                        mt(1:N,1:C,k)=grad;
                        vt(1:N,1:C,k)=grad.^2;
                    else
                        normC(1:N,1:C,k)=min(2+lambda*epoch,4);
                        mt(1:N,1:C,k)=beta1*mt(1:N,1:C,k)+(1-beta1)*grad;
                        vt(1:N,1:C,k)=beta2*vt(1:N,1:C,k)+(1-beta2)*abs(grad).^normC(1:N,1:C,k);
                    end
                    if epoch==1
                        mt1(1:N,k)=del(1:N,k);
                        vt1(1:N,k)=del(1:N,k).^2;
                        
                    else
                        normC1(1:N,k)=min(2+lambda*epoch,4);
                        mt1(1:N,k)=beta1*mt1(1:N,k)+(1-beta1)*del(1:N,k);
                        vt1(1:N,k)=beta2*vt1(1:N,k)+(1-beta2)*abs(del(1:N,k)).^normC1(1:N,k);
                    end
                end
                mth=mt./(1-beta1^epoch);
                vth=vt./(1-beta2^epoch);
                delw=alpha*mth./(vth.^(1./normC)+epsilon);
                mth1=mt1./(1-beta1^epoch);
                vth1=vt1./(1-beta2^epoch);
                delb=alpha*mth1./(vth1.^(1./normC1)+epsilon);
                weights=weights+delw;
                bias=bias+delb;
        end
    end
    [~,cm,~,~] = confusion(tar', outtemp');

    errorRecord(epoch)=e;    
    plot(app.UIAxes,1:length(errorRecord),errorRecord);

    xlim(app.UIAxes,'auto')
    ylim(app.UIAxes,'auto')
    title(app.UIAxes,'Algorithm Performance')
    xlabel(app.UIAxes,'Epoch')
    ylabel(app.UIAxes,'Training Cost')
    drawnow
    if epoch>10 && abs(errorRecord(epoch-1)-e)< goal
        break;
    end
%     app.TrainingRecordTextArea.Value{epoch}= strcat('Epoch: ',num2str(epoch),...
%         ' Error: ',num2str(e),' Success: ',num2str(success), '/',num2str(ninput));
end
trial.layers=config.layers;
%trial.attempt=attempt;
trial.weight=weights;
trial.bias=bias;
trial.record=errorRecord;
trial.error=errorRecord(end);
trial.alpha=alpha;
trial.beta1=beta1;
trial.beta2=beta2;
trial.momentum=momentum;
% if attempt==1
min_error=e;
min_w=weights;
min_b=bias;
min_alpha=alpha;
min_record=errorRecord;
min_beta1=beta1;
min_beta2=beta2;
% elseif e < min_error
%     min_error=e;
%     min_w=weights;
%     min_b=bias;
%     min_alpha=alpha;
%     min_record=errorRecord;
%     min_beta1=beta1;
%     min_beta2=beta2;
% end



% switch algo
%     case 'RMSprop'
%         save('RMSprop_result.mat','trial');
%     case 'AdpRMSprop'
%         save('AdpRMSprop_result.mat','trial');
%     case 'Exp'
%         save('Exp_result.mat','trial');
%     case 'SGD'
%         save('SGD_result.mat','trial');
%     case 'HBMom'
%         save('HBMom_result.mat','trial');
%     case 'Adam'
%         save('Adam_result.mat','trial');
%     case 'AdpAdam'
%         save('AdpAdam_result.mat','trial');
%     case 'AbsDiff'
%         save('AbsDiff_result.mat','trial');
% end
%filename=strcat(algo,num2str(config.layers),'_result.mat');
%save(filename,'trial');
network.weights=min_w;
network.bias=min_b;
network.error=min_record;
network.epochs=epochs;
network.trials=trial;

end
%---------sigmoid.m-----------------------------
function val=sigmoid(x)
[m,n] = size(x);
val = zeros(m,n);
for i =1:m
    for j=1:n
        val(i,j)=(1/(1+(exp(-x(i,j)))));
    end
end
end
function val = tansigmoid(x)
val = tanh(sigmoid(x));
end
function val=tansigmoid1(x)
val=1-tansigmoid(x).^2 .*sigmoid1(x);
end
function val=sigmoid1(x)
val=sigmoid(x).*(1-sigmoid(x));
end
