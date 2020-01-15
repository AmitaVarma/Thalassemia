function [model, mse] = mlp(X, Y, h)
h = [size(X,1);h(:);size(Y,1)];
L = numel(h); %no of hidden layers
W = cell(L-1); %weights
for l = 1:L-1
  W{l} = randn(h(l),h(l+1));
end
Z = cell(L);
Z{1} = X;
eta = 1/size(X,2); %learning rate eta
maxiter = 20000;
mse = zeros(1,maxiter);
for iter = 1:maxiter
  % forward propagation
  for l = 2:L
    Z{l} = sigmoid(W{l-1}'*Z{l-1});
  end

  % backpropagation
  E = Y-Z{L};
  mse(iter) = mean(dot(E(:),E(:)));
  for l = L-1:-1:1
    df = Z{l+1}.*(1-Z{l+1}); %gradient descent
    dG = df.*E;
    dW = Z{l}*dG';
    W{l} = W{l}+eta*dW; %weight-updation
    E = W{l}*dG;
  end
end
mse = mse(1:iter);
model.W = W;
function y = mlpPred(model, X)
W = model.W;
L = length(W)+1;
Z = cell(L);
Z{1} = X;
for l = 2:L
  Z{l} = sigmoid(W{l-1}'*Z{l-1});
end
y = Z{L};
clear; close all;
h = [150,150];
% h: L x 1 vector specify number of hidden nodes in each layer l
% h: 2 x 1, 150 hidden nodes
% X: 27 x 142 (d= no of features =5, n = no of data points= 142)
% T: 5 x 142 (p= no of o/p features =1, n = no of data points= 142)
OrigBase= csvread('C:\Users\Amita\Desktop\Amita\Projects\ANN and fuzzy control\final data.csv',1,0,[1,0,142,0]);
MutaBase= csvread('C:\Users\Amita\Desktop\Amita\Projects\ANN and fuzzy control\final data.csv',1,1,[1,1,142,1]);
Ethnicity=csvread('C:\Users\Amita\Desktop\Amita\Projects\ANN and fuzzy control\final data.csv',1,2,[1,2,142,2]);
Range= csvread('C:\Users\Amita\Desktop\Amita\Projects\ANN and fuzzy control\final data.csv',1,3,[1,3,142,3]);
Zygosity= csvread('C:\Users\Amita\Desktop\Amita\Projects\ANN and fuzzy control\final data.csv',1,4,[1,4,142,4]);
Type= csvread('C:\Users\Amita\Desktop\Amita\Projects\ANN and fuzzy control\final data.csv',1,5,[1,5,142,5]);
OrigBase=dec2base(OrigBase,10)-'0';
MutaBase=dec2base(MutaBase,10)-'0';
Ethnicity=dec2base(Ethnicity,10)-'0';
Range=dec2base(Range,10)-'0';
Zygosity=dec2base(Zygosity,10)-'0';
Type=dec2base(Type,10)-'0';
T=Type;
X=[OrigBase MutaBase Ethnicity Range Zygosity];
X=X';T=T';
[model,mse] = mlp(X,T,h);
r=randi(142,10,1);
avg=0;
for i=1:10
  a=X(:,r(i));
  p=mlpPred(model,a);
  y=T(:,r(i));
  e=y-p;
  avg=avg+abs(e);
end
avg=avg*10;
error=mean(avg);
accuracy=100-error;
