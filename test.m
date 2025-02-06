%%We use the glass dataset to test our FusionkNN
clear;clc;
load('glass.mat')
classnum = 6;
para.lambda_1 =1;
para.lambda_2 =1;
para.lambda_3 =10^-3;
sigma = 1;
sim = 0.7;
X=NormalizeFea(X,0);
[nn,~] = size(X);
for i = 1:nn
    diff_matrix = X(i, :) - X(i, :).';
    squared_diff = diff_matrix .^ 2;
    K(i).k1 = exp(-squared_diff / (2 * sigma^2));
end
clear diff_matrix squared_diff;

ind = crossvalind('Kfold',size(find(Y),1),10);
for k = 1:10
    testindex = ind(:) == k;
    trainindex = ~testindex;
    id = find(trainindex~=0);
    for i2 =1 : length(id)
        Ker(i2).k2 = K([find(trainindex~=0)]).k1;
    end
    [~, ~, W ] = FSKNN(X(trainindex,:),X(testindex,:),Ker,para);
    clear  Ker;
    [time, label ] = KNN_HD( X(trainindex,:),Y(trainindex,:),X(testindex,:),W,sim);
    predY{k}=label;
    Time(k) = time;
end
bb = [];
for tt = 1:10
    aa = predY{tt};
    bb = vertcat(bb,aa);
    aa = [];
end
pr_Y(:,1)= bb;
Acc = Accuracy( pr_Y(:,1),Y );
sumTime = sum(Time);
fprintf('The accuracy is %8.5f\n',Acc)
fprintf('The running cost is %8.5f\n',sumTime)