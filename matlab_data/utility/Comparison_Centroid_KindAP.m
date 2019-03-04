%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparison Centroid based algorithms and Subspace based algorithms on same data.
% Last modified by F.C. 09/05/2017
% Last modified by Y.Y. 02/26/2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ten algorithms are compared:
%      1 - 2.  Kmeans1*   &&    Kmedians1* 
%      3.      Spectral Rotation
%      4 - 5.  KindAP     &&    KindAP+L
%      6.      KindR
%      7.      Joint Model
%      8 - 10. Kmeans  &&  Kmedians  &&  Kmedoids
%--------------------------------------------------------------------------
% Five equalities are stored:
%      1. K-means objective value: fm
%      2. K-indicators objective value: fi
%      3. runnning time: the average time of kmeans and kmedoids 
%      4. Clustering accuracy: ac
%      5. Normalized mutual information: nmi
%      6. Clustering index: idx
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     

fm = zeros(1,10); fi = fm; t = fm; ac = fm; nmi = fm; idx = zeros(n,10);

fgt = Objective_Centers(idxg,k,U);
fprintf('                 fg = %9.6e  generated\n',fgt)

%% ===============================================================
%% Ground Truth: Run Lloyd start from ground truth center
if run_1star == 1
    % K-means with squared Euclidean distance (K-means)
    t0 = tic;
    [~,~,Centers] = Objective_Centers(idxg,k,U(:,1:k));
    idx1 = kmeans(U(:,1:k),k,'Start',Centers,'OnlinePhase',Phase_kmeans,'Distance','sqeuclidean');
    t(1) = toc(t0);
    [fi(1),fm(1)] = Objective_Centers(idx1,k,U);
    idx1 = bestMap(idxg,idx1);
    ac(1) = 100*sum(idxg == idx1)/n;
    nmi(1) = 100*MutualInfo(idxg,idx1);
    idx(:,1) = idx1;
    fprintf('kmeans1*:        fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        fm(1),fi(1),ac(1),nmi(1),t(1))
    
    % K-means with cityblock distance (K-medians)
    t0 = tic;
    [~,~,Centers] = Objective_Centers(idxg,k,U(:,1:k));
    idx2 = kmeans(U(:,1:k),k,'Start',Centers,'OnlinePhase',Phase_kmeans,'Distance','cityblock');
    t(2) = toc(t0);
    [fi(2),fm(2)] = Objective_Centers(idx2,k,U);
    idx2 = bestMap(idxg,idx2);
    ac(2) = 100*sum(idxg == idx2)/n;
    nmi(2) = 100*MutualInfo(idxg,idx2);
    idx(:,2) = idx2;
    fprintf('kmedians1*:      fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
            fm(2),fi(2),ac(2),nmi(2),t(2))
end

if run_SR == 1
    warning off
    t0 = tic;
    min_obj_sr = Inf;
    
% Nie's version: SR
%     for restart=1:No_SR
%         G0 = sparse(1:n,randi(k,n,1),ones(n,1));
%         [~, ~, Gsr, obj_sr] = SRvsKM(U, G0);
%         if obj_sr(end) < min_obj_sr
%             min_obj_sr = obj_sr(end);
%             [~,idx3] = max(Gsr,[],2);
%         end
%     end

% Our version: SR
    options.isnrmrowU = 1; options.binary = 1;options.do_inner = 0;
    for restart = 1:No_SR
        idx3temp = KindAP(U,k,options);
        curr_obj_sr = Objective_Centers(idx3temp,k,U);
        if curr_obj_sr < min_obj_sr
            min_obj_sr = curr_obj_sr;
%             fi(3) = curr_obj_sr;
            idx3 = idx3temp;
        end
    end
    t(3) = toc(t0) / No_SR;
    [fi(3),fm(3)] = Objective_Centers(idx3,k,U);
    idx3 = bestMap(idxg,idx3);
    ac(3) = 100*sum(idxg == idx3)/n;
    nmi(3) = 100*MutualInfo(idxg,idx3);
    idx(:,3) = idx3;
    fprintf('SR%5i:         fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
    No_SR,fm(3),fi(3),ac(3),nmi(3),t(3))
end
    
    % The following lines are commented by Yuchen
%     % K-medoids with squared Euclidean distance (K-mediods)
%     t0 = tic;
%     Centers = Objective_Centers2(idxg,k,U(:,1:k));
%     idx3 = kmedoids(U(:,1:k),k,'Start',Centers,'OnlinePhase',Phase_kmeans);
%     t(3) = toc(t0);
%     [fi(3),fm(3)] = Objective_Centers(idx3,k,U);
%     idx3 = bestMap(idxg,idx3);
%     ac(3) = 100*sum(idxg == idx3)/n;
%     nmi(3) = 100*MutualInfo(idxg,idx3);
%     idx(:,3) = idx3;
%     fprintf('kmedoids1*:      fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
%         fm(3),fi(3),ac(3),nmi(3),t(3))
%     
% end

%% KindAP
if run_kind == 1
    options = [];
    t0 = tic;
    % to make a fair comparison with SR, otherwise comment the following
    options.isnrmrowU = 1; options.binary = 1;
    [idx4,~,~,out] = KindAP(U,k,options); t(4)=toc(t0);
    [fi(4),fm(4)] = Objective_Centers(idx4,k,U(:,1:k));
    idx4 = bestMap(idxg,idx4);
    ac(4) = 100*sum(idxg == idx4)/n;
    nmi(4) = 100*MutualInfo(idxg,idx4);
    idx(:,4) = idx4;
    fprintf('KindAP:          fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        fm(4),fi(4),ac(4),nmi(4),t(4))
end

%% KindAP + Lloyd
if correction == 1
    % KindAP + Kmeans
    % to make the comparison with SR fair
    options.isnrmrowU = 1; options.binary = 1;
    [idx4,~,~,out] = KindAP(U,k,options);
    [~,~,Centers1] = Objective_Centers(idx4,k,U(:,1:k));
    t0 = tic;
    [idx5,~,sumD] = kmeans(U(:,1:k),k,'Start',Centers1,'OnlinePhase',Phase_kmeans,'Distance','sqeuclidean');
    fm(5) = sum(sumD);
    t(5) = toc(t0) + t(4);
    fi(5) = Objective_Centers(idx5,k,U(:,1:k));
    idx5 = bestMap(idxg,idx5);
    ac(5) = 100*sum(idxg == idx5)/n;
    nmi(5) = 100*MutualInfo(idxg,idx5);
    idx(:,5) = idx5;
    fprintf('KindAP+Kms:      fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        fm(5), fi(5),ac(5),nmi(5),t(5))
    
    
    % KindAP + Kmedians
%     if run_kmedians == 1
%         [~,~,Centers1] = Objective_Centers(idx4,k,U(:,1:k));
%         t0 = tic;
%         [idx6,~,sumD] = kmeans(U(:,1:k),k,'Start',Centers1,'OnlinePhase',Phase_kmeans,'Distance','cityblock');
%         fm(6) = sum(sumD);
%         t(6) = toc(t0) + t(4);
%         fi(6) = Objective_Centers(idx6,k,U);
%         idx6 = bestMap(idxg,idx6);
%         ac(6) = 100*sum(idxg == bestMap(idxg,idx6))/n;
%         nmi(6) = 100*MutualInfo(idxg,idx6);
%         idx(:,6) = idx6;
%         fprintf('KindAP+Kns:      fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
%             fm(6), fi(6),ac(6),nmi(6),t(6))
%     end
    
    
    % KindAP + Kmediods
%     if run_kmedoids == 1
%         Centers1 = Objective_Centers2(idx4,k,U(:,1:k));
%         t0 = tic;
%         [idx7,~,sumD] = kmeans(U(:,1:k),k,'Start',Centers1,'OnlinePhase',Phase_kmeans);
%         fm(7) = sum(sumD);
%         t(7) = toc(t0) + t(4);
%         fi(7) = Objective_Centers(idx7,k,U);
%         idx7 = bestMap(idxg,idx7);
%         ac(7) = 100*sum(idxg == bestMap(idxg,idx3))/n;
%         nmi(7) = 100*MutualInfo(idxg,idx7);
%         idx(:,7) = idx7;
%         fprintf('KindAP+Kds:      fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
%             fm(7), fi(7),ac(7),nmi(7),t(7))
%     end
end
%% Joint
if run_joint == 1
    % joint model 02/25/2019 Yuchen
    options = [];
    t0 = tic; 
    % uncomment if internal steps are needed
    % options.disp = 0; options.idxg = idxg;
    idx7 = KindAP_joint(K,k,options); t(7)=toc(t0);
    [fi(7),fm(7)] = Objective_Centers(idx7,k,U(:,1:k));
    idx7 = bestMap(idxg,idx7);
    ac(7) = 100*sum(idxg == idx7)/n;
    nmi(7) = 100*MutualInfo(idxg,idx7);
    idx(:,7) = idx7;
    fprintf('Joint:           fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        fm(7),fi(7),ac(7),nmi(7),t(7))
end

%% KindR
% Last updated 02/25/2019 Yuchen
if run_R == 1
    
    options = [];
    % to make the comparison with SR fair
    options.isnrmrowU = 1; options.binary = 1;
    t0 = tic; 
    [idx6,~,~,out] = KindR(U,k,options); t(6)=toc(t0);
    [fi(6),fm(6)] = Objective_Centers(idx6,k,U(:,1:k));
    idx6 = bestMap(idxg,idx6);
    ac(6) = 100*sum(idxg == idx6)/n;
    nmi(6) = 100*MutualInfo(idxg,idx6);
    idx(:,6) = idx6;
    fprintf('KindR:           fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        fm(6),fi(6),ac(6),nmi(6),t(6))
end


%% Lloyd
% K-means n
if run_kmeans == 1       
    warning('OFF', 'stats:kmeans:FailedToConvergeRep')
    t0 = tic;
    [idx8,~,sumD] = kmeans(U(:,1:k),k,'Replicates',No_kmeans,'OnlinePhase',Phase_kmeans);
    fm(8) = sum(sumD);
    t(8) = toc(t0)/No_kmeans;
    fi(8) = Objective_Centers(idx8,k,U);
    idx8 = bestMap(idxg,idx8);
    ac(8) = 100*sum(idxg == idx8)/n;
    nmi(8) = 100*MutualInfo(idxg,idx8);
    idx(:,8) = idx8;
    fprintf('Kmeans%5i:     fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        No_kmeans,fm(8),fi(8),ac(8),nmi(8),t(8))
end
% K-medians n
if run_kmedians == 1       
    warning('OFF', 'stats:kmeans:FailedToConvergeRep')
    t0 = tic;
    [idx9,~,sumD] = kmeans(U(:,1:k),k,'Replicates',No_kmedians,'OnlinePhase',Phase_kmedians,'Distance','cityblock');
    fm(9) = sum(sumD);
    t(9) = toc(t0)/No_kmedians;
    fi(9) = Objective_Centers(idx9,k,U);
    idx9 = bestMap(idxg,idx9);
    ac(9) = 100*sum(idxg == idx9)/n;
    nmi(9) = 100*MutualInfo(idxg,idx9);
    idx(:,9) = idx9;
    fprintf('Kmedians%5i:   fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        No_kmedians,fm(9),fi(9),ac(9),nmi(9),t(9))
end
% K-medoids n
if run_kmedoids == 1    
    t0 = tic;
    [idx10,~,sumD] = kmedoids(U(:,1:k),k,'Replicates',No_kmedoids,'OnlinePhase',Phase_kmedoids);
    fm(10) = sum(sumD);
    t(10) = toc(t0)/No_kmedoids;
    fi(10) = Objective_Centers(idx10,k,U);
    idx10 = bestMap(idxg,idx10);
    ac(10) = 100*sum(idxg == idx10)/n;
    nmi(10) = 100*MutualInfo(idxg,idx10);   
    idx(:,10) = idx10;
    fprintf('Kmedoids%5i:   fm = %9.6e  fi = %9.6e  AC = %6.2f%%  MI = %6.2f%%  t = %6.2fs\n',...
        No_kmedoids,fm(10),fi(10),ac(10),nmi(10),t(10))   
end
%% ===============================================================
fprintf('                 fg = %9.6e  generated\n',fgt)