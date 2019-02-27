function C = Objective_Centers2(idx,k,U)
%====================================================================
% Computing centers and objectives from given data matrix and index.
% Last modified by F.C. 10/08/2016

%====================================================================
% Input:
% idx: n by 1 cluster indices for data points
%   k: the number of clusters
%   U: n by r data matrix
%====================================================================
% Output:
%  fm: k-means objective value |U - UHH'|_F^2
%  fi: k-indicators objective value min_Z |UZ - H|_F^2 s.t. Z'Z = I
%   C: k by r center matrix 
%   H: n by k indicator matrix
%====================================================================
ind = zeros(1,k);
for j = 1:k, J = find(idx == j);
    if ~isempty(J), 
        len = length(J);
        f = inf; 
        for i = 1:len
            f1 = sum(sum((repmat(U(J(i),:),len,1) - U(J,:)).^2));
            if f1 < f, ind(j) = J(i); f = f1; end
        end
        
    end
end
C = U(ind,:);
end

