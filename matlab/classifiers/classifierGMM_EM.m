function [Ytp,classifier] = classifierGMM_EM(Zs,ks,Zt,classifier)
% (Unsupervised) Expectation maximisation (EM) Gaussian mixture model
% (maximum likelihood estimates)
%
% Inputs 
% Zs = source data
% ks = no. of source components
% Zt = target data
% classifier = pretrained classifier
%
% Outputs
% Ytp = target label predictions
% classifier = trained classifier

if nargin <4
    % Train unsupervised GMM Classifier
    classifier = gmm_mle_em(Zs,ks);
end

% Predict
Ytp = gmm_mle_predict(classifier,Zt);

end
