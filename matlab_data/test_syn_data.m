% test the synethic data for KindAP/KindOT/kmeans/hierarchy
% for ray-shaped, blob-shaped with different number of clusters
% copyright: Yuchen Yang 2018
% initialization
% timing
tc=[];
TC=[];
% accuracy
ac=[];
AC=[];
% outliers dectection
OT=[];
% k is the number of cluster you want to test
for k=10:10:100
    % d is the dimension of constructed data
    d=512;
    % m1,m2 refers to the multiplicity of data clouds
    m1=100;
    m2=100;
    A1=sqrt(2)*randn(k/2,d);
    A2=sqrt(2)*randn(k/2,d);
    % boolean variables
    addray=0;
    addkmeans=0;
    addlink=0;
    addplot=1;
    % real nonnegative variables
    addnoise=1;
    addoutliers=10;
    if addray==0
        %point data
        M=[repmat(A1,m1,1);repmat(A2,m2,1)];
    else
        %ray data;
        M=[kron(randi(100,[m1,1]),A1);kron(randi(100,[m2,1]),A2)];
    end
    % add some perturbation
    if addnoise
        noise=randn(size(M));
        M=M+addnoise*noise;
    end

    % generate ground truth
    idxg=[repmat(1:k/2,1,m1) repmat(k/2+1:k,1,m2)]';
    ts=cputime;
    % prepossessing svd
    [s,D,~]=svd(M,'econ');
    Uk=s(:,1:k+1);
    % call KindAP (my version: kind_ap)
    [idx,~,~,out]=KindAP(Uk,k+1,[]);
    % store time and accuracy
    tc=[tc,cputime-ts];
    ac = [ac sum(idxg == bestMap(idxg,idx(1:end)))/size(M,1)];
    fprintf('k=%d, KindAP finished.\n',k)
    % outlier detection
    if addoutliers
        % add some outliers
        if addray
            M=[M;sqrt(2)*rand(addoutliers,d)*randi(100)];
        else
            M=[M;sqrt(2)*randn(addoutliers,d)];
        end
        [s,~,~]=svd(M,'econ');
        Uk=s(:,1:k);
        % The selection of mu is interesting, depending on n and k
        % Set the last parameter to be 1 if you want detailed results
        [idx,idc]=kind_ot_admm(Uk);
        fprintf('k=%d, outlier-tolerant KindAP finished.\n',k);
        % means of all added outliers, how many of them can be discovered
        ot_recall = length(find(ismember(size(M,1)-addoutliers+1:size(M,1),idc)==1))/addoutliers;
        % means of all detected outliers, how many of them are indeed correct
        ot_acc =  length(find(ismember(idc,size(M,1)-addoutliers+1:size(M,1))==1))/length(idc);
        % means of other normal data, what is the clustering precision
        non_ot_acc = sum(idxg == bestMap(idxg,idx(1:end-addoutliers)))/(size(M,1)-addoutliers);
        OT=[OT,[ot_acc;non_ot_acc;ot_recall]];
    end

    % compare with k-means
    if addkmeans==1
        ts=cputime;
        [IDX,~,~] = kmeans(M,k,'Start','plus','Replicates',10);
        TC = [TC cputime-ts];
        AC = [AC sum(idxg == bestMap(idxg,IDX(1:end-addoutliers)))/(size(M,1)-addoutliers)];
        % csvwrite('pred_kmeans_matlab.csv',IDX)
        fprintf('k=%d, K-means finished.\n',k)
    end
    if addlink==1
        ts=cputime;
        Z=linkage(M);% with or without parameters
        idxlink = cluster(Z,'maxclust',k);
        TC = [TC cputime-ts];
        AC = [AC sum(idxg == bestMap(idxg,idxlink))/size(M,1)];
    end
    %%
    if addplot
        close all
        %M2 = s(:,1:2)*D(1:2,1:2);
        M2 = tsne(M);
        scatter(M2(:,1),M2(:,2),'.');
        hold on
        if addoutliers
            scatter(M2(idc,1),M2(idc,2),'*');
            hold on
            scatter(M2(end-addoutliers+1:end,1),M2(end-addoutliers+1:end,2),'o');
            hold on
            axis off
            legend({'All Data','Detected Outliers','Constructed Outliers'},'FontSize',15)
            title('Synthetic Data with Outliers by ADMM-KindOT','FontSize',20)
        else
            axis off
            legend({'All Data'},'FontSize',15)
            title(sprintf('Synthetic Data by KindAP with accuracy %3.2f',100*ac(end)),'FontSize',20)
        end
    end
   
end