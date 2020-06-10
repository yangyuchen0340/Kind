%% Test on real data sets!
clear; close all
fprintf('****************************************\n')
fprintf('Test on real data sets!\n')

rng('default')

diary('result_Demo_S2Real_data.txt');
diary on;

%% Expermential setting

new_data = 1;

% kmeans setting
run_1star = 1;  run_kind = 1; correction = 1; 
run_SR = 1;  No_SR = 10; 
run_joint = 0; run_R = 0;
run_kmeans = 1;   No_kmeans = 10;    Phase_kmeans = 'on';
run_kmedians = 0; No_kmedians = 10;  Phase_kmedians = 'on';
run_kmedoids = 0; No_kmedoids = 10;  Phase_kmedoids = 'on';

fprintf('run_1star: %i, run_kind: %i, correction: %i\n',...
    run_1star,run_kind,correction)
fprintf('run_kmeans:   %i,   No_kmeans:   %i,   Phase_kmeans:   %s\n',...
    run_kmeans, No_kmeans, Phase_kmeans)
fprintf('run_kmedians: %i,   No_kmedians: %i,   Phase_kmedians: %s\n',...
    run_kmedians, No_kmedians, Phase_kmedians)
fprintf('run_kmedoids: %i,   No_kmedoids: %i,   Phase_kmedoids: %s\n',...
    run_kmedoids, No_kmedoids, Phase_kmedoids)

%% load data
 filename = {
'australian','auto','balance','breast','cars','chess',...,
    'control','crx','dermatology','diabetes','ecoli',...,
    'german','glass','heart','ionosphere','iris','isolet',...,
    'lenses','letter','monk1','pima','segment','solar',...,
    'vehicle','vote','waveform-21','wine','yeast','zoo',...,
    'FERET','JAFFE','USPS','mnist_6000','COIL20',...,
     'Yale_64x64','YaleB_32x32','ORL_64x64','PIE_32x32',...,
     'Reuters21578','TDT2','COIL100','AR'};
% filename = {'australian','auto','balance','breast','cars',...,
%     'control','crx','dermatology','diabetes','ecoli',...,
%     'german','glass','heart','ionosphere','iris','isolet',...,
%     'lenses','monk1','pima','segment','solar',...,
%     'vehicle','vote','waveform-21','wine','yeast','zoo'};
% filename = {'USPS'};
len = length(filename);

T = []; FM =[]; FI =[]; AC = []; NMI = []; S = []; IDX = {}; Ch = cell(1,length(filename));

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
    if size(fea,1)*size(fea,2)<=1e6
        fea = fea - mean(fea);
    end
    Norms = sqrt(sum(fea.^2,2));
    fea =  bsxfun(@times, double(fea), 1./Norms);
    
    
       
    % Similarity matrix setting
    Metric = 'Euclidean';
    NeighborMode = 'KNN'; parameter = 1; knn = 5;
    if strcmp(Metric, 'cosine')
        WeightMode = 'cosine';
    else
        WeightMode = 'HeatKernel';
    end
    
    
    %% Start test
    idxg = gnd; M = double(fea);
    [n,d] = size(M); k = nClass;
    fprintf('\n[n,d,k] = [%i,%i,%i]\n',n,d,k)
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
        [U,~,G] = Eigenspace_generation(K,k,Type);
        toc
    end
    
    %% main
    %%%%%%%%%%%%%%%%%%%%%%%
    Comparison_Centroid_KindAP;
    Ns = sort(out.N,2,'descend');
    s = 1 - Ns(:,2)./Ns(:,1);
    fprintf('Soft indicator: Mean(s) = %.2f \n',mean(s));
    %%%%%%%%%%%%%%%%%%%%%%%
    T  = [T; t]; FM = [FM; fm]; FI = [FI; fi]; %#ok<*AGROW>
    AC = [AC; ac]; NMI = [NMI; nmi]; S = [S; mean(s)];
    Chname = split(filename{j},'_');
    Ch{j} = sprintf(strcat(Chname{1},'&%d'),k);
end

%% Create a table
Rh = {'Algorithms','KindAP','KindAP+L','Kmeans10','SR'};
precision = 3;

outFM = mat2table(FM(:,[4,5,8,3]),Rh,Ch,precision,'min');
disp(outFM)
outAC = mat2table(AC(:,[4,5,8,3]),Rh,Ch,precision,'max');
disp(outAC)
% outNMI = mat2table(NMI(:,[4,5,8,3]),Rh,Ch,precision);
% disp(outNMI)

diary off;
% save result_real_data.mat T FM FI AC NMI
