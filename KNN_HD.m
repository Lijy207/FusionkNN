function [time, label ] = KNN_HD( traindata,Ytrain,testdata,W,sim )
tic;
X_t = traindata;
X_s = testdata;
[m,d] = size(testdata);
W_k = W(:,d+1:d+m);
W_f = W(:,1:d);
total = sum(sum(W_f));
weight = sum(W_f,1);
featureweight = weight./total;
%%
[~, idx2] = sort(abs(featureweight), 'descend');
num = ceil(sim*d);
featureweight(idx2(1:num-1)) = 1;
featureweight(idx2(num:d)) = 0;
idx = find(featureweight == 0);
X_t(:,idx) = [];
X_s(:,idx) = [];
%%
threshold = mean(mean(W_k));
for i = 1:m
    optimal(i) = length(find(W_k(:,i)>threshold));
    Label(i) = knnclassify(X_s(i,:),X_t,Ytrain, optimal(i), 'euclidean', 'nearest');
end

label = Label';
time = toc;


end

