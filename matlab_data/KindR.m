function [idx,H,dUH,out] = KindR(Uk,k,options)
%====================================================================
% Solving K-indicators by Alternating Projection algorithm. (KindAP)
%
% Copyright: Yuchen Yang 2018
% Reference: "Data clustering: K-means versus K-indicators"
% Feiyu Chen, Liwei Xu, Taiping Zhang, Yin Zhang
% Last modified by Y.Z. 09/23/2016
% Copyright: Feiyu Chen, Yuchen Yang, Yin Zhang. 2018
% Last modified by Yuchen Yang. 12/16/2018
%====================================================================
% Input:
% Uk: n by k (in most cases column-orthonormal) matrix
%     (usually k eigenvectors of a Gram or Laplacian matrix)
% k: the number of clusters
% options: a struct for parameters with the fields
%    -- U     : an orthonormal, k-D basis in span(U) [default: Uk]
%    -- tol   : tolerance for inner stopping rule    [default: 1e-3]
%    -- runkm : run kmeans starts with centers of KindAP [default: 0]
%    -- maxit1: maximum iterations for outer iter    [default: 50]
%    -- maxit2: maximum iterations for inner iter    [default: 200]
%    -- idisp : level of iteration info display      [default: 1]
%    -- isnrm : whether to normalize H columnwise    [default: based on Uk]
%    -- doskip : whether to continue without inner iter  [default: 1]
%====================================================================
% Output:
% idx: n by 1 cluster indices for data points
%   H: n by k indicator matrix
% dUH: the objective function value of K-indicators
% out: struct for other output information
%====================================================================

if nargin < 2, k = size(Uk,2); end
if nargin < 3 || isempty(options), options = struct([]); end
if isfield(options,'U'), U = options.U; else, U = Uk(:,1:k); end
if isfield(options,'tol'), tol = options.tol; else, tol = 1e-3; end
if isfield(options,'maxit1'), maxit1=options.maxit1; else, maxit1=50; end
if isfield(options,'maxit2'), maxit2=options.maxit2; else, maxit2=200; end
if isfield(options,'disp'), idisp = options.disp; else, idisp = 0; end
if isfield(options,'runkm'), runkm = options.runkm; else, runkm = 0; end
if isfield(options,'idxg'), idxg = options.idxg; else, idxg = []; end
if isfield(options,'isnrmU'), isnrmrowU = options.isnrmrowU; else, isnrmrowU = 0; end
if isfield(options,'isnrmH'), isnrmcolH = options.isnrmcolH; else, isnrmcolH = ~isnrmrowU; end
if isfield(options,'postSR'), postSR = options.postSR; else, postSR = 1; end
if isfield(options,'do_inner'), do_inner = options.do_inner; else, do_inner = 1; end
if isfield(options,'binary'), binary = options.binary; else, binary = 0; end

[n,~] = size(Uk); 
if isnrmrowU, Uk = normr(Uk);end
idx = ones(n,1); 
hist = zeros(maxit1,1); 
numiter = zeros(maxit1,1);
N = zeros(n,k); H = N; dUH = 2*k;
crit1 = zeros(3,1);crit2 = zeros(4,1);
Z = eye(k);
% Outer iterations:
for Outer = 1:maxit1
    idxp = idx; Up = U; Np = N; Hp = H; ci = 0; 
    %% Step 1: Uo <---> N
    % Inner iterations:
    if do_inner
        [FinalX, fv, ~, ~, iter] = testAP(Uk,Z,tol,maxit2,idisp);
        Z = FinalX.main;
        U = Uk*Z;
        N = max(U,0);
        numiter(Outer) = iter+1;
        hist(Outer) = sqrt(2*fv);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        iter = 0;
        numiter(Outer) = 0;
        N = max(0,U);
    end
    %% Step 2:  N  ---> H
    [idx,H] = Projection_H(N,isnrmcolH,binary);
    idxchg = norm(idx-idxp,1);
    if ~isempty(idxg) && idisp > 1
        AC = 100*sum(idxg==bestMap(idxg,idx))/n; 
    end
    
    %% Step 3:  H  ---> Uo
    [U,Z] = Projection_Uo(H,Uk);
    dUHp = dUH; dUH = norm(U-H,'fro');
    
    %% Check stop condition
    crit2(1) = dUH < sqrt(eps);              % only for ideal case
    crit2(2) = abs(dUHp-dUH) < dUHp*sqrt(eps); % almost no change in dUH
    crit2(3) = dUH > dUHp;                   % distance increases
    crit2(4) = idxchg == 0;                  % no change in clusters
    if idisp
        fprintf('Outer%3i: %3i(%1i)  dUH: %11.8e  idxchg: %6i',...
            Outer,iter,ci(1),dUH,idxchg)
        if ~isempty(idxg) && idisp > 1, fprintf('  AC: %6.2f%%',AC); end
        fprintf('\n')
    end
    if any(crit2)
        if postSR && do_inner, do_inner = 0; binary = 1; continue; end
        if crit2(3) && ~crit2(2), idx=idxp; H=Hp; U=Up; N=Np; dUH=dUHp; end
        if idisp, fprintf('\tstop criteria: (%i,%i,%i,%i)\n',crit2); end
        break;
    end
    
end % Outer iterations

out.C = (bsxfun(@rdivide,Uk'*H,sum(H,1)))'; % Compute the centers
if runkm, idx = kmeans(Uk,k,'Start',out.C); end
out.H = H;
out.U = U;
out.N = N;
out.outer = Outer;
out.numiter = numiter(1:Outer);
out.hist = hist(1:Outer);

end % KindAP

%=====================================================================
%% external functions: Projections
%=====================================================================
function [U,R] = Projection_Uo(N,Uk)
T = Uk' * N;
[Ut,St,Vt] = svd(T);
R = Ut * Vt';
U = Uk * R;
end



%% _________________________________________________
function [idx,H] = Projection_H(N,isnrmcolH,binary)
[n,k] = size(N);
[v,idx] = max(N,[],2);
if binary
    H = sparse(1:n,idx,ones(n,1),n,k);
else
    H = sparse(1:n,idx,v,n,k);
end
if isnrmcolH
    %H = proj_H_normalize(N);
    H = normalize_cols(H); % Normalization
end
end

%% _________________________________________________
function X = normalize_cols(X) %%#ok<DEFNU>
% normalize the columns of matrix X
d = sqrt(sum(X.^2)); d(d==0) = 1;
X = bsxfun(@rdivide,X,d);
end