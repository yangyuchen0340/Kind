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
for k=100:10:100
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
    % real nonnegative variables
    addnoise=1;
    addoutliers=100;
    if addray==0
        %point data
        M=[repmat(A1,m1,1);repmat(A2,m2,1)];
    else
        %ray data;
        M=[kron(A1,randi(100,[m1,1]));kron(A2,randi(100,[m2,1]))];
    end
    % add some perturbation
    if addnoise
       noise=randn(size(M))-0.5;
       M=M+addnoise*normr(noise);
    end

    % generate ground truth
    idxg=[repmat(1:k/2,1,m1) repmat(k/2+1:k,1,m2)]';
    ts=cputime;
    % prepossessing svd
    [s,~,~]=svd(M,'econ');
    Uk=s(:,1:k);
    % call KindAP (my version: kind_ap)
    [~,idx]=kind_ap(Uk,0,0);
    % store time and accuracy
    tc=[tc,cputime-ts];
    ac = [ac sum(idxg == bestMap(idxg,idx(1:end)))/size(M,1)];
    fprintf('k=%d, KindAP finished.\n',k)
    % outlier detection
    if addoutliers
        % add some outliers
        M=[M;sqrt(2)*rand(addoutliers,d)];
        [s,~,~]=svd(M,'econ');
        Uk=s(:,1:k);
        [idx,idc]=kind_ot_admm(Uk,0.01,1,0);
        fprintf('k=%d, outlier-tolerant KindAP finished.\n',k);
        % means of all added outliers, how many of them can be discovered
        ot_recall = length(find(ismember(size(M,1)-addoutliers+1:size(M,1),idc)==1))/addoutliers;
        % means of all detectd outliers, how many of them are indeed correct
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
        AC = [AC sum(idxg == bestMap(idxg,IDX))/size(M,1)];
        AC_1 = [AC_1 sum(idxg(1:m1*k/2) == bestMap(idxg(1:m1*k/2),IDX(1:m1*k/2)))/(m1*k/2)];
        % csvwrite('pred_kmeans_matlab.csv',IDX)
    end
    if addlink==1
        ts=cputime;
        Z=linkage(M);% with or without parameters
        idxlink = cluster(Z,'maxclust',k);
        TC = [TC cputime-ts];
        AC = [AC sum(idxg == bestMap(idxg,idxlink))/size(M,1)];
    end
    
   
end