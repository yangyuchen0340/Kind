%% Expermential setting
clear; close all
fprintf('****************************************\n')
fprintf('Test on real data sets!\n')

rng('default')

diary('result_d_greater_than_k_data.txt');
diary on;
new_data = 1; Type = 'Normalized Laplacian'; add_plot = 0;


% kmeans setting
run_1star = 0;    run_kind = 1; correction = 1; 
run_SR = 1;  No_SR = 10;
run_R = 0; run_joint = 1;
run_kmeans = 1;   No_kmeans = 10;    Phase_kmeans = 'on';
run_kmedians = 0; No_kmedians = 1e+2;  Phase_kmedians = 'on';
run_kmedoids = 0; No_kmedoids = 1e+2;  Phase_kmedoids = 'on';

fprintf('run_1star: %i, run_kind: %i, correction: %i\n',...
    run_1star,run_kind,correction)
fprintf('run_kmeans:   %i,   No_kmeans:   %i,   Phase_kmeans:   %s\n',...
    run_kmeans, No_kmeans, Phase_kmeans)
fprintf('run_kmedians: %i,   No_kmedians: %i,   Phase_kmedians: %s\n',...
    run_kmedians, No_kmedians, Phase_kmedians)
fprintf('run_kmedoids: %i,   No_kmedoids: %i,   Phase_kmedoids: %s\n',...
    run_kmedoids, No_kmedoids, Phase_kmedoids)


filename = {'australian','auto','balance','breast','cars',...,
    'control','crx','dermatology','diabetes','ecoli',...,
    'german','glass','heart','ionosphere','iris','isolet',...,
    'lenses','monk1','pima','segment','solar',...,
    'vehicle','vote','waveform-21','wine','yeast','zoo'};
filename = {'catsndogs_4000_vgg16'};
len = length(filename);

T = []; FM =[]; FI =[]; AC = []; NMI = []; S = []; IDX = {}; KID = [];

for j=1:len
    fprintf('****************************************\n')
    fprintf( strcat(filename{j},'\n'))
    fprintf('****************************************\n')
    load (strcat(filename{j},'.mat'));
    
     % Adjust names and format
    if size(gnd,1)<size(gnd,2)
        gnd = gnd';
    end
    if exist('alls','var')
        fea = alls'; 
        clear alls
    end
    
    
    % Normalization
    Norms = sqrt(sum(fea.^2,2));
    fea =  bsxfun(@times, double(fea), 1./Norms);
%     fea = fea - mean(fea);
    
       
    % Similarity matrix setting
    Metric = 'Euclidean'; NeighborMode = 'KNN';
    WeightMode = 'HeatKernel'; parameter = 1; knn = 5;
    
    
%     if j >= 8
%         Metric = 'cosine'; NeighborMode = 'KNN';
%         knn = 5; WeightMode = 'cosine'; parameter = 1;
%     end
    
    
    %% Start test
    idxg = gnd; M = double(fea);
    [n,d] = size(M);
    nClass = length(unique(gnd)); max_k = min(d,3*nClass);
    k = nClass;
    fprintf('\n[n,d,k] = [%i,%i,%i]\n',n,d,k);
    if new_data == 1
        %% Similarity Matrix generation
        options = [];
        options.Metric = Metric; options.NeighborMode = NeighborMode;
        options.k = knn; options.WeightMode = WeightMode;
        options.t = parameter; options.gnd = idxg;
        tic      
        K = constructW(M,options); K = (K + K')/2;
        toc
        %% Eigenspace_generation
        Type = 'Normalized Laplacian';
        tic
        %U = Eigenspace_generation(K,k,Type);
        [U0,~,G] = Eigenspace_generation(K,max_k,Type);
        toc
    end
    

%% Run
    for s = k:max_k
        fprintf('---------------------------------\n')
        fprintf('Test on %d features with %d clusters\n',s,k);
        [Q,~] = qr(randn(s));
        U = U0(:,1:s)*Q;
        
        Comparison_Centroid_KindAP; 
        T  = [T; t]; FM = [FM; fm]; FI = [FI; fi];
        AC = [AC; ac]; NMI = [NMI; nmi]; 
    end
    KID = [KID;max_k];
%% Plot
    if add_plot
        Y = tsne(U);
        close all
        plot(Y(:,1),Y(:,2),'.')
    end
end

save result_d_greater_than_k_data.mat T FM FI AC NMI