function X=prox_l2(T,mu)
% Solve soft thresholding with l2 norm
% original: argmin_{X in R^{n*k}} 1/2||X-T||_2^2+mu||X||_{1,2}
% row-wise: argmin_{x in R^k} 1/2||x-t||_2^2+mu||x||2
% Reference: A fast algorithm for edge-preserving variational multichannel 
% image restoration  (Lemma 3.3)
% Author: Yuchen Yang
    X = zeros(size(T));
    for i=1:size(T,1)
        t=T(i,:);
        nrm=norm(t,2);
        if nrm<mu
            x=zeros(size(t));
        else
            x=(1-mu/nrm)*t;
        end
        X(i,:)=x;
    end
end