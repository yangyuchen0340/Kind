function M = points_inside_sphere(c,r,n)
%====================================================================
% Uniformly place n points inside a sphere with center c and radius r.
% Last modified by F.C. 10/08/2016

%====================================================================
% Input:
% c: 1 by d center vector of sphere
% r: radius of sphere
% n: number of points inside sphere.
%====================================================================
% Output:
% M : n by d data matrix.
%====================================================================

d = size(c,2);
N = randn(n,d); Norms = sum(N.^2,2);
% The incomplete gamma function is used to map N radially inside the
% sphere with a uniform spatial distribution.
M = repmat(c,n,1) + N.*repmat(r*(gammainc(Norms/2,d/2).^(1/d))./sqrt(Norms),1,d);

end