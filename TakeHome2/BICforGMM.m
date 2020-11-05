function BICforGMM
% Generates N samples from a specified Gaussian Mixture PDF
% then uses EM algorithm to estimate the parameters along 
% with BIC to select the model order, which is the number 
% of Gaussian components for the model.

close all,

N = 1000; % Number of samples

% Generate samples from a 3-component Gaussian Mixture PDF
% This is the true data pdf...
alpha = [0.2,0.3,0.5];
mu = [-10 0 10;0 0 0]';
Sigma(:,:,1) = [3 1;1 20];
Sigma(:,:,2) = [7 1;1 2];
Sigma(:,:,3) = [4 1;1 16];
gmtrue = gmdistribution(mu,Sigma,alpha);
[d,Mtrue] = size(gmtrue.mu'); % determine dimensionality of samples and number of GMM components
x = random(gmtrue,N)'; % Draw N iid vector samples from the specified GM pdf

nSamples = d*N; % number of scalar data values, though not independent, counting as if they are
% Evaluate BIC for candidate model orders
maxM = floor(N^(1/2)); % arbitrarily selecting the maximum model using this rule
for M = 1:maxM
    M,
    nParams(1,M) = (M-1) + d*M + M*(d+nchoosek(d,2));
    % (M-1) is the degrees of freedomg for alpha parameters
    % d*M is the derees of freedomg for mean vectors of M Gaussians
    % M*(d+nchoosek(d,2)) is the degrees of freedom in cov matrices
    % For cov matrices, due to symmetry, only count diagonal and half of
    % off-diagonal entries.
    options = statset('MaxIter',1000); % Specify max allowed number of iterations for EM
    % Run EM 'Replicates' many times and pickt the best solution
    % This is a brute force attempt to catch the globak maximum of
    % log-likelihood function during EM based optimization
    gm{M} = fitgmdist(x',M,'Replicates',20,'Options',options); 
    neg2logLikelihood(1,M) = -2*sum(log(pdf(gm{M},x')));
    BIC(1,M) = neg2logLikelihood(1,M) + nParams(1,M)*log(nSamples);
    figure(1), plot([1:M],BIC(1:M),'.'), 
    xlabel('Number of Gaussian Components in GMM'),
    ylabel('BIC'),
    drawnow,
end
[~,bestM] = min(BIC),
bestGMM = gm{bestM},
keyboard,

%%% You can modify this code to visuaize all trained GMMs
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 

