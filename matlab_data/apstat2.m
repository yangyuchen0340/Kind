% comparison with kmeans
% d=512; multiplicity=100
ac=[];
tc=[];
ac_1=[];
TC=[];
AC=[];
AC_1=[];
for k=200:10:200
d=512;
m1=10;
m2=100;
A1=sqrt(2)*randn(k/2,d);% zeros(k,d-k)];
A2=sqrt(2)*randn(k/2,d);
% A=randn(k,d);
addray=0;
addnoise=1;
addkmeans=0;
addlink=0;
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
idxg=[repmat(1:k/2,1,m1) repmat(k/2+1:k,1,m2)]';
ts=cputime;
[s,~,~]=svd(M,'econ');
Uk=s(:,1:k);
[err,idx,~,H]=ap2(Uk,0,0);
tc=[tc,cputime-ts];
ac = [ac sum(idxg == bestMap(idxg,idx))/size(M,1)];
%ac_1 = [ac_1 sum(idxg(1:m1*k/2) == bestMap(idxg(1:m1*k/2),idx(1:m1*k/2)))/(m1*k/2)];
%apstat(Uk,200,1);
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
% cross validation by python
IDX1=csvread('pred_kmeans.csv');
IDX1(1)=[];