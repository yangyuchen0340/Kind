%% Test on real data sets!
clear; close all
fprintf('****************************************\n')
fprintf('Test on real data sets with outliers!\n')

rng('default')

%% Expermential setting

new_data = 1;

% kmeans setting
run_kind = 1; correction = 0; 
run_kmeans = 1; No_kmeans = 10;    Phase_kmeans = 'off';
run_od = 1;

fprintf('run_od: %i, run_kind: %i, correction: %i\n',...
    run_od,run_kind,correction)
fprintf('run_kmeans:   %i,   No_kmeans:   %i,   Phase_kmeans:   %s\n',...
    run_kmeans, No_kmeans, Phase_kmeans)

%% load data
%  filename = {
%     'australian','auto','balance','breast','cars','chess',...,
%     'control','crx','dermatology','diabetes','ecoli',...,
%     'german','glass','heart','ionosphere','iris','isolet',...,
%     'lenses','letter','monk1','pima','segment','solar',...,
%     'vehicle','vote','waveform-21','wine','yeast','zoo',...,
%     'FERET','JAFFE','USPS','mnist_6000','COIL20',...,
%      'Yale_64x64','YaleB_32x32','ORL_64x64','PIE_32x32','Reuters21578','TDT2','AR'};
filename = {'australian','auto','balance','breast','cars',...,
    'control','crx','dermatology','diabetes','ecoli',...,
    'german','glass','heart','ionosphere','iris','isolet',...,
    'lenses','monk1','pima','segment',...,
    'vehicle','vote','waveform-21','wine','yeast','zoo'};
len = length(filename);

T = []; FM =[]; FI =[]; AC = []; NMI = []; PC = []; RC = []; IDX = {};

for j = 1:len
    
    fprintf('****************************************\n')
    fprintf( strcat(filename{j},'\n'))
    fprintf('****************************************\n')
    load (strcat(filename{j},'.mat'));
    
    % Adjust names and format
    if size(gnd,1)<size(gnd,2)
        gnd = gnd';
    end
    gnd = double(gnd);
    if exist('alls','var')
        fea = alls'; 
        clear alls
    end
   
    nClass = length(unique(gnd));
    % Normalization
    fea = fea - mean(fea);
    Norms = sqrt(sum(fea.^2,2));
    fea =  bsxfun(@times, double(fea), 1./Norms);
%     

    
       
    % Similarity matrix setting
    Metric = 'Euclidean'; NeighborMode = 'KNN';
    WeightMode = 'HeatKernel'; parameter = 1; knn = 5;
    
    %% Outliers Split and Start the test
    k = round(0.8*nClass);
    unique_gnd = unique(gnd); unique_gnd = unique_gnd(randperm(length(unique_gnd))); %unique_gnd = [1,2,5,6,8,3,4,7];
    remained_samples = []; remained_gnd = [];
    for kk = 1:k
        remained_samples = [remained_samples; fea(gnd == unique_gnd(kk),:)];
        remained_gnd = [remained_gnd; kk*ones(sum(gnd == unique_gnd(kk)),1)];% gnd(gnd == unique_gnd(kk))];
    end
    outliers_samples = [];
    for kk = k+1:nClass
        temp = datasample(find(gnd == unique_gnd(kk)),1);
        outliers_samples = [outliers_samples; fea(temp,:)];
    end
    outliers_gnd = -1*ones(size(outliers_samples,1),1);
    
    all_data = [remained_samples, remained_gnd; outliers_samples, outliers_gnd];
%     all_data = all_data(randperm(size(all_data,1)),:);
    gnd_ot = double(all_data(:,end)); M = double(all_data(:,1:end-1));
    idxg = gnd_ot(gnd_ot>0);
    [n,d] = size(M); ot = length(outliers_gnd);
    fprintf('\n[n,d,nClass,k,ot] = [%i,%i,%i,%i,%i]\n',n,d,nClass,k,ot);
    
%% Similarity Matrix generation
    if new_data == 1
        options = [];
        options.Metric = Metric; options.NeighborMode = NeighborMode;
        options.k = knn; options.WeightMode = WeightMode;
        options.t = parameter; 
        
        
        tic     
        K = constructW(M,options); K = (K + K')/2;
        % self-tuning kernels
%         neighbor_num = 8;  D = dist2(M,M); 
%         [~,K,~] = scale_dist(D,neighbor_num); K = (K + K')/2; 
        % fixed kernels for small n
%         D = dist2(M,M); scale = 0.8; K = exp(-D/(scale^2)); K = (K + K')/2;
        toc
        %% Eigenspace_generation
        Type = 'Normalized Laplacian';
        tic
        %U = Eigenspace_generation(K,k,Type);
        U = Eigenspace_generation(K,k,Type);
        toc
    end
    %% main
    %%%%%%%%%%%%%%%%%%%%%%%
    Run_KindOT;
    %%%%%%%%%%%%%%%%%%%%%%%
    T  = [T; t]; FM = [FM; fm]; FI = [FI; fi]; %#ok<*AGROW>
    AC = [AC; ac]; NMI = [NMI; nmi]; PC = [PC; pc]; RC = [RC; rc];
    
end
