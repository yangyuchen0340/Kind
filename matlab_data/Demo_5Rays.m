%% Test on rays data
clear; close all
fprintf('****************************************\n')
fprintf('Test on rays data\n')
fprintf('****************************************\n')
addpath utility
%rng('default')


run_kind = 1; run_kmeans = 1;
No_kmeans = 5; Phase_kmeans = 'on'; Normalize = 0;

%% K-rays data generation

d = 100; n = 1800; k = 6; ndk = n/k;
% Centers and Weights
C = randn(k,d); W = 3*abs(randn(ndk,k));
idxg=reshape(repmat(1:k,n/k,1),n,1); % ground truth index.

% data
M = [];
for i = 1:k
    M = [M ; kron(W(:,i),C(i,:))]; %#ok<AGROW>
end
N = randn(size(M));
M = M + 1e-1*norm(M,'fro')/norm(N,'fro')*N;

if Normalize == 1
    Norms = sqrt(sum(M'.^2,1));
    M = (bsxfun(@times, M', 1./Norms))';
end
[Ut,St,Vt] = svd(M); U = Ut(:,1:k);

%% KindAP
if run_kind == 1
    options.disp = 1;options.doskip=0;
    [idx,~,~,out] = KindAP(U,k,options);
    Ns = sort(out.N,2,'descend');
    s = 1 - Ns(:,2)./Ns(:,1);
    fprintf('Soft indicator: Mean(s) = %.2f \n',mean(s));
    idx = bestMap(idxg,idx);
    AC = 100*sum(idxg == bestMap(idxg,idx))/n;
end

%% Kmeans
if run_kmeans == 1
    idxk = kmeans(U,k,'Replicates',No_kmeans,'OnlinePhase',Phase_kmeans);
    idxk = bestMap(idxg,idxk);
    ACk = 100*sum(idxg == bestMap(idxg,idxk))/n;
end

%% Plot figures
dotsize = 8;
colormap([1 0 1;   % magenta
    1 0 0;   % red
    0 0 .8;   % blue
    0 .6 0;   % dark green
    .3 1 0;   % bright green
    1 0.8 0]);   % orange


subplot(131)
scatter(M(:,1), M(:,2), dotsize, idxg);
set (gca,'fontsize',15)
xlabel('First Dimension','FontSize',20)
ylabel('Second Dimension','FontSize',20)
title('Original Data','FontSize',20)

subplot(132)
scatter(M(:,1), M(:,2), dotsize, idxk);
set (gca,'fontsize',15)
xlabel('First Dimension','FontSize',20)
ylabel('Second Dimension','FontSize',20)
title('Result of Lloyd','FontSize',20)

subplot(133)
scatter(M(:,1), M(:,2), dotsize, idx);
set (gca,'fontsize',15)
xlabel('First Dimension','FontSize',20)
ylabel('Second Dimension','FontSize',20)
title('Result of KindAP','FontSize',20)
shg
