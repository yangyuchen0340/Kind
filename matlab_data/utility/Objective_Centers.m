function [fi,fm,C,H] = Objective_Centers(idx,k,U)
%====================================================================
% Computing centers and objectives from given data matrix and index.

% version 1.2 --11/29/2018
% version 1.1 --11/19/2018
% version 1.0 --10/08/2016

% Feiyu Chen

%====================================================================
% Input:
% idx: n by 1 cluster indices for data points
%   k: the number of clusters
%   U: n by r data matrix
%====================================================================
% Output:
%  fm: k-means objective value 1/2*||U - UHH'||_F^2 = k - ||U'H||_F^2
%  fi: k-indicators objective value k - ||U'H||_*
%   C: k by r center matrix 
%   H: n by k indicator matrix
%====================================================================

n = length(idx); 
H = sparse(1:n,idx,ones(n,1),n,k); 

H0 = H; % added Yuchen
U0 = normr(U);% added Yuchen

H = normalize_cols(H); % Normalization 
s = svd(U'*H,0);  % si is the i-th singular value (Principle angle) of U'*H.


C = (bsxfun(@rdivide,U' * H,sum(H,1)))'; 

% fi = 2*(k - sum(s)); % F.C.
s0 = svd(U0'*H0,0);% added Yuchen
fi = trace(H0'*H0) + trace(U0'*U0) - 2*sum(s0); % Y.Y 11/28/2018
%fm = k - sum(s.^2);  % F.C. 11/19/2018
fm = k + trace(U'*U) - sum(s.^2); % added Yuchen

% sumds = zeros(k,1);
% for j = 1:k, J = find(idx == j);
%     if ~isempty(J), sumds(j) = sum(var(U(J,:),1)*length(J)); end
% end
% fm = sum(sumds);
end
%=====================================================================
%% external function
%=====================================================================
function X = normalize_cols(X) %%#ok<DEFNU>
% normalize the columns of matrix X
d = sqrt(sum(X.^2)); d(d==0) = 1; 
X = bsxfun(@rdivide,X,d);
end