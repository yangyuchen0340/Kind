function [X,d,G] = Eigenspace_generation(M,k,options)

% Given a data matrix M \in R^{n x d}, this function generates the
% smallest/largest k eigenvectors (X \in R^{n x k} of some kernal matrices.
% Gram: MM'
% Laplacian: L = D - K, where, and D = diag(sum(K,2)).
% Normalized Laplacian: L = I - D^{-1/2}*K* D^{-1/2}

if nargin < 3
    options = 'Gram';
end

[n,m] = size(M);

r = min( min(n,m), k);

switch lower(options)
    case {lower('Gram')}
        if m > 2*n
            K = M*M'; K = (K + K')/2;
            G = K;
            [X,D] = eigs(K,r);
            d = diag(D);
            [d,id] = sort(d,'descend');
            X = X(:,id);
            %             X = sqrt(D)*V';
        elseif m == k
            K = M'*M;  K = (K + K')/2;
            G = K;
            [U,D] = eig(K);
            d = diag(D);
            [d,id] = sort(d,'descend');
            U = U(:,id);
            H = M*U;
            X = H*diag(1./sqrt(d));
        else
            K = M'*M;  K = (K + K')/2;
            G = K;
            [V,D] = eigs(K,r);
            d = diag(D);
            [d,id] = sort(d,'descend');
            V = V(:,id);
            H = M*V;
            X = H*diag(1./sqrt(d));
        end   
        
    case{lower('Normalized Laplacian')}
        K = M;

        D = 1./sqrt(sum(K,1));
        G = D'.*K.*D;
        L = speye(n) - G; L = (L+L')/2;
        warning('OFF','MATLAB:eigs:SigmaChangedToSA')
        [V,Diag] = eigs(L,k,'sr');

        %D = 1./sqrt(sum(K,1)); nd = numel(D);
        %DNH = sparse(1:nd,1:nd,D);
        %L = speye(n) - DNH*K*DNH; L = (L+L')/2;
        %G = speye(n) - L;
        %warning('OFF','MATLAB:eigs:SigmaChangedToSA')
        %[V,Diag] = eigs(L,k,'sr');

         d = diag(Diag);
%         [d,id] = sort(d,'ascend');  V = V(:,id);
        V = V*diag(sign(double(max(V)==max(abs(V)))-0.5)); if max(max(V))<0; V = -V; end;     
        X = real(V);
    otherwise
        error('Selection does not exist!');      
        
end
X = real(X); d = real(d);
end
