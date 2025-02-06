function [ alpha, beta, W ] = FSKNN( traindata, testdata, Ker, para )
%traindata represents the training data
%testdata represents the testing data
X_t = traindata;   %Training data
X_s = testdata;     %test data
lambda_1 = para.lambda_1;  %parameter lambda_1
lambda_2 = para.lambda_2;     %parameter lambda_1
lambda_3 = para.lambda_3;   %parameter lambda_3
[n,d] = size(X_t);
[m,~] = size(X_s);


%% Perform Gaussian kernel mapping
% sigma = 1;
% for i = 1:n
%     diff_matrix = X_t(i, :) - X_t(i, :).';
%     squared_diff = diff_matrix .^ 2;
%     K(i).k = exp(-squared_diff / (2 * sigma^2));
% end

classnumber = 2;  % Number of groups
%% Group by fuzzy c-means clustering
index = zeros(classnumber,n);
[~,U,~] = fcm(X_t,classnumber); % U is the membership matrix
IGg = [];
for ii = 1:classnumber
    indexx = [];
    indexx =  find(U(ii,:) > 0.01);
    index(ii,1:length(indexx)) = indexx;
    Ig = zeros(1,n);
    Ig(sub2ind(size(Ig), indexx)) = 1;
    IGg(ii,:) = Ig; %Constructing IGg matrix
end
%% initialization
W = rand(n,d+m);
beta = rand(n,1);
Wsum = sum(W,2);
% beta = ones(n,1);

nt = 10^3;
iter = 1;
beta2(1) = 1;
iii = 1;
WW(iter).w = beta;
obji = 1;
t=1;
L = 0.3;
while 1
    clear D;
    for i = 1:n
        Groupsum = 0;
        for ii = 1:classnumber
            indexx = [];
            indexx =  find(U(ii,:) > 0.01);
            group = [Wsum(indexx)];
            Groupsum = Groupsum+(IGg(ii,i)*norm(group,1))/norm(W(i,:),1);
        end
        ff(i) = Groupsum;
    end
    D = diag(ff);
    for i =1:n
        dn(i) = sqrt(sum((sum(W.*W,2)+eps)))./sum(W(i,:));
    end
    P = diag(dn);
    sumK = zeros(d,d);
    for ii = 1:n
        sumK = sumK+ beta(ii).*Ker(ii).k2;
    end
    KK = [sumK;X_s];
    V = (X_t*X_t'+ lambda_1.*P + lambda_2.*D).^(-1/2)*X_t*KK';
    [M,S,N] = svds(V);
    W = (X_t*X_t'+ lambda_1.*P + lambda_2.*D).^(-1/2)*M*N';
    %% update alpha
    alpha = trace(X_t'*W*KK)/trace(KK'*KK);
    %% update beta

    objv_new=10^12;
    objv=10^12+1;
    while ~(objv-objv_new<0.1||t>10)
        objv =  norm(W(:,1:d)'*X_t - alpha.*sumK, 'fro') + lambda_3*sum(abs(beta));

        sbeta = [];
        for j = 1:d
            for j1 = 1:n
                Kbeta = Ker(j1).k2;
                sbeta(j1,:) = Kbeta(j,:);
            end
            Sbeta(j).s =  sbeta;
            dF1(:,j) = 2*alpha^2*sbeta*sbeta'*beta - 2*alpha*sbeta*X_t'*W(:,j);
        end
        dF = sum(dF1,2);

        swap=objv_new;
        objv_new=objv;
        objv=swap;
        z=beta-dF./L;
        pre_beta=beta;
        for i=1:n
            if lambda_3/L<z(i)
                beta(i)=z(i)-lambda_3/L;
            elseif abs(z(i))<=lambda_3/L
                beta(i)=0;
            elseif z(i)<-lambda_3/L
                beta(i)=z(i)+lambda_3/L;
            end
        end
        if objv_new>objv&&objv_new<10000
            beta=pre_beta;
            break
        end
        t=t+1;

    end

    iter = iter+1;
    if (iter > 5)    break,     end

end
clear sumK Ker Sbeta;

end

