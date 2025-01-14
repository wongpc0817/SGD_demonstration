function output = applynetwork(network,Inputs)
switch network.model
    case 'MLP'
        % Format:
        %   network.weights = MxMxN matrix, referring to the weights on each layer
        %   network.struct = Lx2 matrix, L = number of layers
        %   inputs = SxQ matrix, S = number of datasets, Q = number of inputs
        %   output = SxU matrix, U = number of outputs
        bias = network.bias;
        layer = network.struct;
        noutput = layer(end,1);
        output = zeros(size(Inputs,1),noutput);
        weights = network.weights;
        noflayers = size(weights,3)+1;
        maximum = max(layer);
        ninput = size(Inputs,1);
        maximum = max([ninput,maximum]);
        intemp=Inputs;
        for k=1:noflayers-1
            N=layer(k,1);C=layer(k,2);
            out=[bias(1:N,k),weights(1:N,1:C,k)]*[ones(1,size(intemp,1));intemp'];
            intemp=sigmoid(out)';
        end
        output=intemp';
    case 'RBF'
        layers = network.struct;
        noutput = layers(end,1);
        w = network.weights;
        noflayers = size(layers,1)+1;
        ninput = size(Inputs,1);
        spread=network.spread;
        c=network.centers;
        x=Inputs;
        for i_x=1:ninput
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
                    intemp(i_x,:)=(w(1:n,1:n+1,layer-1)*[1;intemp(i_x,:)'])';
                    for l=1:N
                        H(l,i_x,layer)=gaussian(intemp(i_x,:),c(l,1:n,layer),spread(layer,l));
                    end
                    outtemp(i_x,1:N,layer)=H(1:N,i_x,layer)';
                end
                n=N;
            end
        end
        output=outtemp(:,1:N,end)';

end
end

function val = sigmoid(x)
[m,n] = size(x);
val = zeros(m,n);
for i = 1:m
    for j= 1:n
        val(i,j) = 1/(1+exp(-x(i,j)));
    end
end
end
function val= tansigmoid(x)
[m,n] = size(x);
for i=1:m
    for j=1:n
        val(i,j)=tanh(1/(1+(exp(-x(i,j)))));
    end
end
end
function val=gaussian(x,c,sigma)
%Returning a real value
val = exp(-sum((x-c).^2) /(sigma^2));
end
function y = classify(x)
t=1:10;
x=5*x+5;
t=abs(t-x);
[~,t]=min(t,[],2);
if size(t,2)>1
    y=t(1);
else
    y=t;
end
end