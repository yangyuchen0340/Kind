%% Test for soft indicator!
clear; close all
fprintf('****************************************\n')
fprintf('Test for soft indicator!\n')
addpath real_data 
addpath utility
rng('default')
colormap jet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Synthetic Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data generation

k = 3; n = 3000; 
c = [-1 0; 1 0; 0 sqrt(3)];
m1 = points_inside_sphere(c(1,:),1,n/3);
m2 = points_inside_sphere(c(2,:),1,n/3);
m3 = points_inside_sphere(c(3,:),1,n/3);
idxg = [ones(n/3,1); 2*ones(n/3,1); 3*ones(n/3,1)]; % ground truth index
data = [m1; m2; m3];

%% KindAP
sigma = .01;
U = run_kernel(data(:,1:2),k,sigma);
[idx,~,~,out] = KindAP(U,k);
idx = bestMap(idxg,idx);
ACi = 100*sum(idxg==idx)/n;
Ns = sort(out.N,2,'descend');
s = 1 - (Ns(:,2)./Ns(:,1));

%% Plot fig.3a
dotsize = 12;
subplot(1,2,1);
scatter(data(:,1), data(:,2), dotsize, s); axis equal; axis([-2 2 -1 1+sqrt(3)]); colorbar
title('A: Synthetic data [n,d,k] = [7500,2,3]')
set(gca,'fontsize',16)
drawnow 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ORL Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Expermential setting

new_data = 1;
% kmeans setting
run_1star = 1; run_kind = 1; correction = 1;
run_kmeans = 0; No_kmeans = 10000;  Phase_kmeans = 'on';
run_kmedoids = 0; No_kmedoids = 10000;  Phase_kmedoids = 'on';

fprintf('run_1star: %i, run_kind: %i correction: %i\n',...
    run_1star,run_kind,correction)
fprintf('run_kmeans: %i, No_kmeans: %i, Phase_kmeans: %s\n',...
    run_kmeans, No_kmeans, Phase_kmeans)

%% load data

load ORL_64x64.mat
IND = [    
27    22    33    25
22     5    39    12
26    25    30    39
13    28    30     6
3     1    23    16];

FEA = fea;
fea = double(fea);
nClass = length(unique(gnd));
% Normalization
Norms = sqrt(sum(fea.^2,2));
fea =  bsxfun(@times, fea, 1./Norms);

T = []; FM =[]; FI =[]; AC = []; NMI = []; IDX = {};

S = zeros(40,5); A = zeros(5,1);
%% Start test
str = (1:5);
k = 4;
for j = str
    dataId = []; gndd = []; FIG = [];
    %     ind=randperm(nClass); dataId = []; gndd = []; ind(1:4)
    for jj = 1:k                            % k is the number of classes.
        id = find(gnd ==IND(j,jj))';
        dataId =[dataId id];
        gndd = [gndd; jj*ones(size(id,2),1)];
        FIG = [FIG; reshape(FEA(gnd == (IND(j,jj)),:)',64,64*10)];
    end
    M = double(fea(dataId,:)); idxg = gndd;    
    
    [n,d] = size(M);
    fprintf('\n[n,d,k] = [%i,%i,%i]\n',n,d,k)
    if new_data == 1
        %% Similarity Matrix generation
        
        options = [];
        options.Metric = 'Euclidean'; options.NeighborMode =  'KNN';
        options.k = 5; options.WeightMode = 'HeatKernel';
        options.t = 1; options.gnd = gndd;
        tic
        K = constructW(M,options); K = (K + K')/2;
        toc
%         figure; imshow(FIG,[]); title(sprintf('Group %i',j))
%         figure; imagesc(K); title(sprintf('Similarity matrix: K%i',j))
        %% Eigenspace_generation
        Type = 'Normalized Laplacian';
        tic
        U = Eigenspace_generation(K,k,Type);
        toc
    end
    
    %% main
    %%%%%%%%%%%%%%%%%%%%%%%
    Comparsion_KindAP_Lloyd;
    Ns = sort(out.N,2,'descend');
    s = 1 - Ns(:,2)./Ns(:,1);    
    S(:,j) = sort(s,'descend'); A(j) = ac(2);
    IDX2 = [idx2, s];
    %%%%%%%%%%%%%%%%%%%%%%%
    T  = [T; t]; FM = [FM; fm]; FI = [FI; fi]; %#ok<*AGROW>
    AC = [AC; ac]; NMI = [NMI; nmi];
    
end

Lstr = char(5,37);
for i = str
    Lstr(i,1:37) = sprintf('Group %i: Mean(s) = %.2f; AC = %6.2f%%', i, mean(S(:,i)),A(i));
end

%% Plot fig.3b
subplot(1,2,2)
h = plot(S(:,str),'*-'); axis([-1 42 0 1])
legend(Lstr(str,:),'location','southwest')
% ylabel('Soft Indicators')
title(sprintf('B: ORL data [n, d, k] = [%i, %i, %i]',n,d,k))
set(gca,'YAxisLocation','right')
set(h,'linewidth',2);
set(gca,'fontsize',16)
grid, shg

