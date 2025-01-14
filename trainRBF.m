function network = trainRBF(app,config)
% Format: config.inputs = SxQ matrix
%         config.targets = SxU matrix
%         config.goal = stopping condition (relative error)
%         config.epochs = iterations
%         config.layers = 1xN matrix, each element refers to the
%         number of hidden neurons on that layer.
%         config.gradAlgo = Gradient Search Algorithm Used Here
%         network.weights = MxMxN matrix
%         network.model = 'RBF'
%         network.epochs = epochs needed
%         network.layers = config.layers 1xN matrix
%         network.gradAlgo = Gradient Search Algorithms Used Here
%         network.struct = layers [Ni,Ci];
%         network.bias = 1xN matrix
%S = number of datasets,
%Q = number of inputs per dataset
%U = number of outputs
%M = maximum number of neurons
%N = number of layers
network.model = 'RBF';
goal = config.goal;
noflayers = size(config.layers,2)+2; %including input and output layers
epochs = config.epochs;
x = config.inputs;
network.trainsize = size(x,1);
tar = config.targets;
algo = config.gradAlgo;
% elesize=size(x,2);
noutput=size(tar,2);
p = randperm(size(x,1));
x = x(p,:);
tar=tar(p,:);
for i=1:noflayers-1
    if i==1
        layers(i,2)=size(x,2);
        layers(i,1)=config.layers(i);
    elseif i==noflayers-1
        layers(i,2)=config.layers(i-1);
        layers(i,1)=noutput;
    else
        layers(i,2)=config.layers(i-1);
        layers(i,1)=config.layers(i);
    end
end
network.struct=layers;
maximum= max(config.layers);
maximum = max([maximum,size(x,2),noutput]);
ninput=size(x,1);
H=zeros(maximum,ninput,noflayers-1);
eta=config.alpha;
mom=0.1;
errorRecord=0;
success=0;
w=2*rand(maximum,maximum+1,noflayers-2)-1;% weight matrix
delb=zeros(size(w(:,1,:)));
spread=zeros(noflayers-1,maximum);
outtemp=zeros(ninput,maximum,noflayers-1);

%initialise hyperparameters
for layer=1:noflayers-1
    N=layers(layer,1); n=layers(layer,2);
    if layer==1
        intemp=x;
    else
        intemp=outtemp(:,1:n,layer-1);
    end
    if layer~=noflayers-1 && layer~=1
        for i_x=1:ninput
            intemp(i_x,:)=(w(1:n,1:n+1,layer-1)*[1;intemp(i_x,:)'])';
        end
    end
    [C,s]=rbfpara(intemp,N);
    c(1:N,1:n,layer)=C;
    spread(layer,1:size(s,2))=s+1e-8;
    for i_x=1:ninput
        if layer==noflayers-1
            
            phi=[1;intemp(i_x,:)'];
            outtemp(i_x,1:N,layer)=(w(1:N,1:n+1)*phi)';%sigmoid((w(1:N,1:n+1)*phi)');%  outtemp=size(x,1) x N
            
        elseif layer==1
            
            for l=1:N
                H(l,i_x,layer)=gaussian(intemp(i_x,:),c(l,1:n,layer),spread(layer,l));
            end
            outtemp(i_x,1:N,layer)=H(1:N,i_x,layer)';
            
        else
            for l=1:N
                H(l,i_x,layer)=gaussian(intemp(i_x,:),c(l,1:n,layer),spread(layer,l));
            end
            outtemp(i_x,1:N,layer)=H(1:N,i_x,layer)';
        end
    end
    n=N;
end
delc=zeros(size(c));
dels=zeros(size(spread));
delw=zeros(size(w));
mtc=zeros(size(delc));
mts=zeros(size(dels));
vtc=zeros(size(delc));
vts=zeros(size(dels));
mtw=zeros(size(delw));
vtw=zeros(size(delw));
mtb=zeros(size(w,1),1);
vtb=zeros(size(mtb));
for epoch =1:epochs
    e=0;
    errorRecord = 0;
    success = 0;
    pdelc = delc;
    pdels = dels;
    pdelw = delw;
    %propagation
    for i_x = 1:ninput
        for layer=1:noflayers-1
            N=layers(layer,1);n=layers(layer,2);
            if layer==1
                intemp=x;
            else
                intemp=outtemp(:,1:n,layer-1);
            end
            
            if layer==noflayers-1
                phi=[1;intemp(i_x,:)'];
                %                     outtemp(i_x,1:N,layer)=sigmoid(w(1:N,1:n+1,layer-1)*phi)';
                outtemp(i_x,1:N,layer)=(w(1:N,1:n+1,layer-1)*phi)';
                
            elseif layer==1
                for l=1:N
                    H(l,i_x,layer)=gaussian(intemp(i_x,:),c(l,1:n,layer),spread(layer,l));
                end
                outtemp(i_x,1:N,layer)=H(1:N,i_x,layer)';
                
            else
                intemp(i_x,1:n)=(w(1:n,1:n+1,layer-1)*[1;intemp(i_x,:)'])';
                Htemp(i_x,1:n,layer-1)=intemp(i_x,1:n);
                for l=1:N
                    H(l,i_x,layer)=gaussian(intemp(i_x,:),c(l,1:n,layer),spread(layer,l));
                end
                outtemp(i_x,1:N,layer)=H(1:N,i_x,layer)';
            end
            n=N;
        end
        %calculate error
        e=e+sum((tar(i_x,:)-outtemp(i_x,1:size(tar,2),end)).^2)/2;
        [~,I]=max(outtemp(i_x,1:size(tar,2),end));
        [~,label]=max(tar(i_x,:));
        if I==label
            success=success+1;
        end
        %error back-propagation
        for k=noflayers-1:-1:1
            N=layers(k,1);n=layers(k,2);
            if k==noflayers-1
                del(1:N,k)=(tar(i_x,:)-outtemp(i_x,1:N,k))';%.*sigmoid1(outtemp(i_x,1:N,k)');
                delw(1:N,2:n+1,k-1)=del(1:N,k)*outtemp(i_x,1:n,k-1);
                delb(1:N,1,k-1)=del(1:N,k);
                for i=1:n
                    del(i,k-1)=sum(del(1:N,k).*w(1:N,i+1,k-1));
                end
            elseif k==1
                for i=1:N
                    delc(i,1:n,k)=del(i,k)*outtemp(i_x,i,k)...
                        .*(x(i_x,1:n)-c(i,1:n,k))./spread(k,i)^2 *(2);
                    dels(k,i)=del(i,k)*outtemp(i_x,i,k)...
                        *sum((x(i_x,1:n)-c(i,1:n,k)).^2) /spread(k,i)^3 *(2);
                end
            else
                for i=1:N
                    delc(i,1:n,k)=del(i,k)*outtemp(i_x,i,k)...
                        .*(Htemp(i_x,1:n,k-1)-c(i,1:n,k))./spread(k,i)^2 *(2);
                    dels(k,i)=del(i,k)*outtemp(i_x,i,k)...
                        *sum((Htemp(i_x,1:n,k-1)-c(i,1:n,k)).^2) /spread(k,i)^3 *(2);
                    delT(i,1:n)=del(i,k)*outtemp(i_x,i,k)*...
                        (-2).*(Htemp(i_x,1:n,k-1)-c(i,1:n,k))./spread(k,i)^2;
                end
                for i=1:n
                    temp(i,1)=sum(delT(:,i));
                end
                
                delb(1:n,1,k-1)=temp(1:n,1);
                delw(1:n,2:n+1,k-1)=temp(1:n,1)*outtemp(i_x,1:n,k-1);
                
                for i=1:n
                    %                         dot(temp(1:n,1),w(1:n,i+1,k-1)')
                    del(i,k-1)=dot(temp(1:n,1),w(1:n,i+1,k-1)');
                end
            end
        end

        I=i_x;
        switch algo
            case 'SGD'
                delc=eta*delc+mom*pdelc;
                dels=eta*dels+mom*pdels;
                delw(:,2:end,:)=eta*delw(:,2:end,:)+eta*mom*pdelw(:,2:end,:);
                delw(:,1,:)=eta*delb(:,1,:)+eta*mom*pdelw(:,1,:);
            case 'HBMom'
                if epoch~=1 || i_x~=1
                    mtc = (mtc.*delc >=0);
                    mts = (mts.*dels >=0);
                    mtw = (mtw.*delw >=0);
                    mtb = (mtb.*delb >=0);
                else
                    mtc=zeros(size(delc));
                    mts=zeros(size(spread));
                    mtw=zeros(size(delw));
                    mtb=zeros(size(delb));
                end
                delc=eta*delc+mom*mtc.*pdelc;
                dels=eta*dels+mom*mts.*pdels;
                delw(:,2:end,:)=eta*delw(:,2:end,:)+eta*mom*mtw(:,2:end,:).*pdelw(:,2:end,:);
                delw(:,1,:)=eta*delb+eta*mom*mtb.*pdelw(:,1,:);
            case 'Adam'
                beta1=0.9;
                beta2=0.999;
                epsilon=1e-8;
                if epoch==1 && i_x==1
                    mtc=delc; mts=dels; mtw=delw; mtb=delb;
                    vtc=delc.^2; vts=dels.^2; vtw=delw.^2; vtb=delb.^2;
                end
                mtc = beta1*mtc+(1-beta1)*delc;
                mtch = mtc./(1-beta1^epoch);
                vtc = beta2*vtc+(1-beta2)*delc.^2;
                vtch = vtc./(1-beta2^epoch);
                delc=eta*mtch./(sqrt(vtch)+epsilon);
                
                mts = beta1*mts+(1-beta1)*dels;
                mtsh = mts./(1-beta1^epoch);
                vts = beta2*vts+(1-beta2)*dels.^2;
                vtsh = vts./(1-beta2^epoch);
                dels=eta*mtsh./(sqrt(vtsh)+epsilon);
                
                mtw = beta1*mtw+(1-beta1)*delw;
                mtwh = mtw./(1-beta1^epoch);
                vtw = beta2*vtw+(1-beta2)*delw.^2;
                vtwh = vtw./(1-beta2^epoch);
                delw(:,2:end,:)=eta*mtwh(:,2:end,:)./(sqrt(vtwh(:,2:end,:))+epsilon);
                
                mtb = beta1*mtb + (1-beta1)*delb;
                mtbh = mtb./(1-beta1^epoch);
                vtb = beta2*vtb+(1-beta2)*delb.^2;
                vtbh=vtb./(1-beta2^epoch);
                delw(:,1,:)=eta*mtbh./(sqrt(vtbh)+epsilon);
            case 'AbsAdam'
                beta1=0.9;
                beta2=0.999;
                epsilon=1e-8;
                if epoch==1 && i_x==1
                    mtc=delc; mts=dels; mtw=delw; mtb=delb;
                    vtc=abs(delc); vts=abs(dels); vtw=abs(delw); vtb=abs(delb);
                end
                mtc = beta1*mtc+(1-beta1)*delc;
                mtch = mtc./(1-beta1^epoch);
                vtc = beta2*vtc+(1-beta2)*abs(delc);
                vtch = vtc./(1-beta2^epoch);
                delc=eta*mtch./(sqrt(vtch)+epsilon);
                
                mts = beta1*mts+(1-beta1)*dels;
                mtsh = mts./(1-beta1^epoch);
                vts = beta2*vts+(1-beta2)*abs(dels);
                vtsh = vts./(1-beta2^epoch);
                dels=eta*mtsh./(sqrt(vtsh)+epsilon);
                
                mtw = beta1*mtw+(1-beta1)*delw;
                mtwh = mtw./(1-beta1^epoch);
                vtw = beta2*vtw+(1-beta2)*abs(delw);
                vtwh = vtw./(1-beta2^epoch);
                delw(:,2:end,:)=eta*mtwh(:,2:end,:)./(sqrt(vtwh(:,2:end,:))+epsilon);
                
                mtb = beta1*mtb + (1-beta1)*delb;
                mtbh = mtb./(1-beta1^epoch);
                vtb = beta2*vtb+(1-beta2)*abs(delb);
                vtbh=vtb./(1-beta2^epoch);
                delw(:,1,:)=eta*mtbh./(sqrt(vtbh)+epsilon);
            case 'ExpAdam'
                beta1=0.9;
                beta2=0.999;
                epsilon=1e-8;
                if epoch==1 && i_x==1
                    mtc=delc; mts=dels; mtw=delw; mtb=delb;
                    vtc=exp(abs(delc)); vts=exp(abs(dels)); vtw=exp(abs(delw)); vtb=exp(abs(delb));
                end
                mtc = beta1*mtc+(1-beta1)*delc;
                mtch = mtc./(1-beta1^epoch);
                vtc = beta2*vtc+(1-beta2)*exp(abs(delc));
                vtch = vtc./(1-beta2^epoch);
                delc=eta*mtch./log(vtch+epsilon);
                
                mts = beta1*mts+(1-beta1)*dels;
                mtsh = mts./(1-beta1^epoch);
                vts = beta2*vts+(1-beta2)*exp(abs(dels));
                vtsh = vts./(1-beta2^epoch);
                dels=eta*mtsh./log(vtsh+epsilon);
                
                mtw = beta1*mtw+(1-beta1)*delw;
                mtwh = mtw./(1-beta1^epoch);
                vtw = beta2*vtw+(1-beta2)*exp(abs(delw));
                vtwh = vtw./(1-beta2^epoch);
                delw(:,2:end,:)=eta*mtwh(:,2:end,:)./log(vtwh(:,2:end,:)+epsilon);
                
                mtb = beta1*mtb + (1-beta1)*delb;
                mtbh = mtb./(1-beta1^epoch);
                vtb = beta2*vtb+(1-beta2)*exp(abs(delb));
                vtbh=vtb./(1-beta2^epoch);
                delw(:,1,:)=eta*mtbh./log(vtbh+epsilon);
        end
        network.weights=w;
        network.spread=spread;
        network.centers=c;
        %Weight updation
        w=w+delw;
        c=c+delc;
        spread=spread+dels;
    end
    %         for i_x = 1:size(x,1)
    %             for layer=1:clusters
    %                 J(layer,i_x)=gaussian(x(i_x,:),c(layer,:),spread(layer));
    %             end
    %             psi(:,i_x)=[1;J(:,i_x)];
    %         end
    %         out=(w*psi)';
    [~,cm,~,~] = confusion(tar', outtemp(1:i_x,1:noutput,end)');
    errorRecord = errorRecord + e/size(x,1);
    errors(epoch)=errorRecord;
    plot(app.UIAxes,1:length(errors),errors);

    xlim(app.UIAxes,'auto')
    ylim(app.UIAxes,'auto')
    title(app.UIAxes,'Algorithm Performance')
    xlabel(app.UIAxes,'Epoch')
    ylabel(app.UIAxes,'Training Cost')
    drawnow
    if epoch~=1 && abs(errors(epoch)-errors(epoch-1))/errors(epoch-1) < goal
        break
    end
% app.TrainingRecordTextArea.Value{epoch}= strcat('Epoch: ',num2str(epoch),...
%         ' Error: ',num2str(e),' Success: ',num2str(success), '/',num2str(ninput));    
end
result.weights=w;
result.spread=spread;
result.centers=c;
result.eta=eta;
result.mom=mom;
result.struct=layers;
result.errorRecord=errors;
network.weights=w;
network.spread=spread;
network.centers=c;
network.struct=layers;
network.trial=result;
network.error=errors;
network.epochs=epoch;
end


function y=gaussian(x,c,sigma)
y = exp(-sum((x-c).^2) /(sigma^2));
end
function val=sigmoid(x)
val=1./(1+exp(-x));
end
function y = sigmoid1(x)
y = x.*(1-x);
end
function [c,s]=rbfpara(x,clusters)
[id,C]=kmeans(x,clusters);
% C=zeros(clusters,elesize);
% for i=1:clusters
%     C(i,:)=x(randi(size(x,1)),:);
% end

% delw=zeros(noutput,clusters);
d_max=norm(C(1,:)-C(2,:));
for i = 1:clusters
    for j =1:clusters-i
        tem = norm(C(i,:)-C(i+j,:));
        if tem >= d_max
            d_max=tem;
        end
    end
end
%sigma1
% spread = zeros(1,clusters) + d_max/sqrt(2*clusters);
%sigma2
num=zeros(1,clusters);
spread=zeros(1,clusters);
for i = 1:size(x,1)
    for j=1:clusters
        if id(i)==j
            spread(1,j)=spread(1,j)+norm(C(j,:)-x(i,:));
            num(j)=num(j)+1;
        end
    end
end
spread = spread./num;
% spread=ones(1,clusters);
% %sigma3
% r = 2*d_max;
% spread=zeros(clusters);
% for j = 1:clusters
%     for i=1:size(x,1)
%         if norm(x(i,:)-C(j,:))<=r
%             spread(j)=spread(j)+ norm(x(i,:)-C(j,:));
%         end
%     end
% end
% spread= spread/r;
% %sigma4
% spread=sum(spread)/clusters;
c=C;
s=spread;
end
