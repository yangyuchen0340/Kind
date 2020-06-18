function [X,mu]=prox_l2_adaptive(T,x)
X = zeros(size(T));
n = size(T,1);
if nargin ==2 
    rownrms = sqrt(sum(T.^2,2));
    val = sort(rownrms,'descend');
    mu = val(x);
else
    if nargin == 1 && n>2
        rownrms = sqrt(sum(T.^2,2));
        val = sort(rownrms,'descend');
        fd2 = val(1:n-2)+val(3:n)-2*val(2:n-1);
        [~,maxid] = max(fd2);
        mu = val(min(maxid+1,floor(0.1*n)));
    else
        error('Invalid input parameter');
    end
end
mu=mu*0.9999;
for i=1:size(T,1)
    t=T(i,:);
    nrm=norm(t,2);
    if nrm>=mu && nrm>0
        X(i,:)=(1-mu/nrm)*t;
    end
end
end