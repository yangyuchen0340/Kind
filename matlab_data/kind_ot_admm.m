function [idx,idc,itr,err,H,V,W,L,Z]=kind_ot_admm(Uk,mu,rho,prt)
%====================================================================
% Solving K-indicators by Alternating Projection algorithm with 
% Outlier-tolerant based on ADMM. (KindAP)
%
% Copyright: Yuchen Yang, Yin Zhang. 2018
% Reference: Stephen Boyd, Neal Parikh, Eric Chu Borja Peleato and Jonathan Eckstein
% Distributed Optimization and Statistical Learning via the 
% Alternating Direction Method of Multipliers. 
%====================================================================
% Input:
% Uk: n by k column-orthonormal matrix
%     (usually k eigenvectors of a Gram or Laplacian matrix)
% mu: the parameter of strength of outliers
% rho: the parameter of augmented Lagrangian dual
% prt: whether printing the residual curve or not
%====================================================================
% Output:
% idc: indices detected as outliers
% idx: cluster indices for data points
% itr: number of iterations
% err: the objective function value of outlier-tolerant K-indicators
%====================================================================
% The optimization problem is formulated as follows:
% argmin_{H,Z,V,W} 1/2||WZ-H||_F^2+mu||V||_{2,1}
% s.t. W,H,V in R^{n*k} (the same shape as give orthonormal matrix Uk) 
%      Z in R^{k*k}, Z^Z=I, H^H=I, H>=0
%      W-V-Uk=0, W^TW=I
%====================================================================
    % initialization
    max_itr = 50;  
    e_abs = 1e-6;
    e_rel = 1e-4;
    err = [];
    [n,k]=size(Uk);
    V = zeros(n,k);
    W = Uk;
    W_old = zeros(n,k);
    L = zeros(n,k);
    for itr=1:max_itr
        % minimize over H,Z by KindAP
        [~,index,~,H,Z]=kind_ap(W,0,0);
        B=W-Uk;
        % minimize V by soft thresholding
        V=prox_l2(B+L/rho,2*mu/rho);
        % minimize W by Procrustes subproblem
        A=V+Uk;
        [s,~,v]=svd(rho*A-L+H*Z',0);
        W=s*v';
        % update dual variable L
        L=L+rho*(W-A);
        S=-rho*(W-W_old);
        W_old=W;
        % save the objectives at each step
        err = [err;0.5*norm(W-H*Z','fro')^2+mu*sum(vecnorm(V,2,2))+trace(L'*(W-A))+rho/2*norm(W-A,'fro')^2];
        % stopping criteria
        prim_res = norm(W-A,'fro');
        dual_res = norm(S,'fro');
        prim_f = prim_res <= sqrt(n*k)*e_abs+max(k,norm(V,'fro'))*e_rel;
        dual_f = dual_res <= sqrt(n*k)*e_abs+norm(L,'fro')*e_rel;
        if prim_f && dual_f
            break;
        end
        % update rho
        if prim_res > 10*dual_res
            rho=rho*2;
        end
        if prim_res < 0.1*dual_res
            rho=rho/2;
        end
    end
    % get the outliers
    idc = find(max(abs(V),[],2)~=0);
    idx = index;
%     idx = index(max(abs(V),[],2)==0);
%     [~,id] = sort(sum(abs(Uk-W),2),'descend');
%     idc = id(1:10);
%     idx = index(id(11:end));
    % plot
    if prt
        figure;
        plot(err,'-r');
        title('The objective value curve','FontSize',20);
    end
    