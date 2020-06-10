function [idx,H,U,out]=KindAP_joint(K,k,options)
% joint clustering
% optimize: max U^T(G+rho*H*H^T)U
% last modified: Yuchen Yang 2/18/19
if isfield(options,'maxit'), maxit=options.maxit1; else, maxit=50; end
if isfield(options,'idxg'), idxg = options.idxg; else, idxg = []; end
if isfield(options,'initidx'), initidx = options.initidx; else, initidx = []; end
if isfield(options,'disp'), idisp = options.disp; else, idisp = 0; end
if isfield(options,'tol'), tol = options.tol; else, tol = 1e-5; end
if isfield(options,'type'), type = options.type; else, type='Normalized Laplacian'; end
if isfield(options,'Phase_kmeans'), Phase_kmeans = options.Phase_kmeans; else, Phase_kmeans='on'; end

n = length(K);
[V,~,G] = Eigenspace_generation(K,k,type);
history = zeros(maxit,1);
% [V,G] = generateV(K,k,3);
if ~isempty(initidx)
    idx=initidx;
else
    idx = KindAP(V,k);
    [~,~,Centers] = Objective_Centers(idx,k,V);
    idx = kmeans(V,k,'Start',Centers,'OnlinePhase',Phase_kmeans);
end
%rho = 0.1;
rho = 1/n;
iter = 1;
while(iter<=maxit)
    Vp = V; idxp = idx;
    H = sparse(1:n,idx,ones(n,1),n,k,n);
    obj = 0.5*(rho*norm(V'*H,'fro')^2+trace(V'*G*V));
    history(iter) = obj;
    if ~isempty(idxg)
        AC = 100*sum(idxg==bestMap(idxg,idx))/n;
        if idisp > 1, fprintf('\t iter: %4d,  AC: %6.2f%%, obj: %6.2f\n',iter, AC, obj); end
    end 
    [V,~] = eigs(@(x)helper(x,H,G,rho),n,k);
    % V=normr(V);
    V = V*diag(sign(double(max(V)==max(abs(V)))-0.5)); if max(max(V))<0; V = -V; end
    idx = KindAP(V,k);
    [~,~,Centers] = Objective_Centers(idx,k,V);
    idx = kmeans(V,k,'Start',Centers,'OnlinePhase',Phase_kmeans);
    Vrel = norm(V-Vp,'fro')/norm(V,'fro'); idxchg = norm(idxp-idx,1);
    if idisp > 0, fprintf('iter: %4d,  Vrel: %6.2e, idxchg: %10d\n',iter, Vrel, idxchg); end
    if Vrel<tol || idxchg == 0
        break;
    end
    iter = iter + 1;
end
U=V;
out.iter = iter-1;
out.history = history(1:iter-1);
end
    
            
function y=helper(x,H,G,rho)
    y=rho*H*(H'*x)+G*x;
end

% function [V,G]=generateV(W,k,Type)
%     % calculate degree matrix
%     n = length(W);
%     degs = sum(W, 2);
%     D    = sparse(1:size(W, 1), 1:size(W, 2), degs);
% 
%     % compute unnormalized Laplacian
%     L = D - W;
% 
%     % compute normalized Laplacian if needed
%     switch Type
%         case 2
%             % avoid dividing by zero
%             degs(degs == 0) = eps;
%             % calculate inverse of D
%             D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
% 
%             % calculate normalized Laplacian
%             L = D * L;
%         case 3
%             % avoid dividing by zero
%             degs(degs == 0) = eps;
%             % calculate D^(-1/2)
%             D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
% 
%             % calculate normalized Laplacian
%             L = D * L * D;
%     end
% 
%     % compute the eigenvectors corresponding to the k smallest
%     % eigenvalues
%     diff   = eps;
%     [U, ~] = eigs(L, k, diff);
% 
%     % in case of the Jordan-Weiss algorithm, we need to normalize
%     % the eigenvectors row-wise
%     if Type == 3
%         U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
%     end
%     V=U;
%     G=speye(n)-L;
% end