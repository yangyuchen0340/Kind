%% Expermential setting
clear; close all
fprintf('****************************************\n')
fprintf('Test on real data sets!\n')

rng('default')

diary('result_DSC_data.txt');
diary on;
new_data = 1; Type = 'Normalized Laplacian'; add_plot = 0;
max_trial = 2;

% kmeans setting
run_1star = 0;    run_kind = 1; correction = 1; 
run_SR = 1;  No_SR = 10;
run_R = 0; run_joint = 0;
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

T = []; FM =[]; FI =[]; AC = []; NMI = []; IDX = {};

fileid = {'ORL','B',20,100};
filename = {'ORL','EYale_B','COIL20','COIL100'};
for j=1:length(fileid)
    id = fileid{j};
    switch id
        case 100
            load CKSym_x100_U.mat
            load gnd_x100.mat
        case 20
            load CKSym_x20.mat
            load gnd_x20.mat
        case 'B'
            load CKSym_xEYaleB.mat
            Label = reshape(repmat((1:38),64,1),2432,1);
        case 'ORL'
            load CKSym_xORL.mat
            load gnd_xORL.mat
    end

    fprintf('****************************************\n')
    fprintf( strcat(filename{j},'\n'))
    fprintf('****************************************\n')   
    n = length(Label);
    k = length(unique(Label));
    fprintf('\n[n,k] = [%i,%i]\n',n,k);
    idxg = Label; 
    if size(Label,1) == 1
        idxg = Label';
    end
    
    if id ~= 100
        run_joint = 1;
        K = W;
        tic
        U = Eigenspace_generation(K,k,Type);
        toc
    else
        run_joint = 0;
        U = W;
    end
    %% Construct U
    % this construction will lead to minor difference in accuracy
%     if size(W,1) == size(W,2)
% 
%         % calculate degree matrix
%         degs = sum(W, 2);
%         D    = sparse(1:size(W, 1), 1:size(W, 2), degs);
% 
%         % compute unnormalized Laplacian
%         L = D - W;
% 
%         % compute normalized Laplacian if needed
%         switch Type
%             case 2
%                 % avoid dividing by zero
%                 degs(degs == 0) = eps;
%                 % calculate inverse of D
%                 D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
% 
%                 % calculate normalized Laplacian
%                 L = D * L;
%             case 3
%                 % avoid dividing by zero
%                 degs(degs == 0) = eps;
%                 % calculate D^(-1/2)
%                 D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
% 
%                 % calculate normalized Laplacian
%                 L = D * L * D;
%         end
% 
%         % compute the eigenvectors corresponding to the k smallest
%         % eigenvalues
%         diff   = eps;
%         [U, ~] = eigs(L, k, diff);
% 
%         % in case of the Jordan-Weiss algorithm, we need to normalize
%         % the eigenvectors row-wise
%         if Type == 3
%             U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
%         end
% 
%     else
%         if size(W,2) == k
%             U=W;
%         else
%             error('invalid input')
%         end
   % end
%% Run
    for trial = 1:max_trial
        fprintf('---------------------------------\n')
        fprintf('Test %d \n',trial);
        [Q,~] = qr(randn(k));
        U = U*Q;
        Comparison_Centroid_KindAP; 
        T  = [T; t]; FM = [FM; fm]; FI = [FI; fi];
        AC = [AC; ac]; NMI = [NMI; nmi];
    end
    
%% Plot
    if add_plot
        Y = tsne(U);
        close all
        plot(Y(:,1),Y(:,2),'.')
    end
end
