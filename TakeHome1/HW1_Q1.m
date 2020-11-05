num_samples = 10000;
gmmParameters.priors = [0.7, 0.3];
p = [0.7, 0.3];
gmmParameters.meanVectors = ones(4,2);
gmmParameters.meanVectors(:, 2) = -1;
gmmParameters.covMatrices = ones(4,4,2);
c_0 = [
    [2, -0.5, 0.3, 0]
    [-0.5, 1,-0.5, 0]
    [0.3, -0.5, 1, 0]
    [0, 0, 0, 2]
    ];
c_1 = [
    [1 0.3 -0.2 0]
    [0.3 2 0.3 0]
    [-0.2 0.3 1 0]
    [0 0 0 3]
    ];
gmmParameters.covMatrices(:,:,1) = c_0;
gmmParameters.covMatrices(:,:,2) = c_1;

[x, labels] = generateDataFromGMM(num_samples, gmmParameters);
Nc = [length(find(labels==1)),length(find(labels==2))];


pxgivenl = ones(2, num_samples);
for l = 1:2
    % Calculate P(x|L=l)
    pxgivenl(l,:) = evalGaussianPDF(x, gmmParameters.meanVectors(:, l), gmmParameters.covMatrices(:,:, l));
end

likelihood_ratios = pxgivenl(2, :) ./ pxgivenl(1, :);
tau = sort(likelihood_ratios(likelihood_ratios >= 0));

for i=1:length(tau)
    decision = likelihood_ratios >= tau(i);
    decision = decision + 1;
    pFA(i) = sum(decision == 2 & labels == 1) / Nc(1);
    pCD(i) = sum(decision == 2 & labels == 2) / Nc(2);
    pE(i) = pFA(i) * p(1) + (1-pCD(i))*p(2);
end
[min_error, min_index] = min(pE);
min_decision = likelihood_ratios >= tau(min_index);
min_FA = pFA(min_index); min_CD = pCD(min_index);

% Find theoretical minimum error
ideal_decision = (likelihood_ratios > p(1)/p(2));
ideal_decision = ideal_decision + 1;
ideal_pFA = sum(ideal_decision == 2 & labels ==1) / Nc(1);
ideal_pCD = sum(ideal_decision == 2 & labels == 2) / Nc(2);
ideal_error = ideal_pFA * p(1) + (1 - ideal_pCD) * p(2);

figure(2); plot(pFA, pCD, '-', min_FA, min_CD, 'o', ideal_pFA, ideal_pCD, 'g+');
set(gca,'fontname','Linux Libertine')
legend('ROC Curve', 'Empirical Min Error', 'Theoretical Min Error');
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title("ROC Curve for ERM Classifier");
saveas(gcf, 'q1_a_roc.png');



% Question 1 Part B
Sigma_NB(:, :, 1) = eye(4); Sigma_NB(:, :, 2) = eye(4);
pxgivenl_nb = ones(2, num_samples);
for l = 1:2
    pxgivenl_nb(l,:) = evalGaussianPDF(x, gmmParameters.meanVectors(:, l), Sigma_NB(:,:, l));
end
likelihood_ratios_nb = pxgivenl_nb(2,:) ./ pxgivenl(1,:);
tau_nb = sort(likelihood_ratios_nb(likelihood_ratios_nb >= 0));
for i = 1:length(tau)
    decision_NB = likelihood_ratios_nb >= tau_nb(i);
    decision_NB = decision_NB + 1;
    pFA_NB(i) = sum(decision_NB == 2 & labels == 1) / Nc(1);
    pCD_NB(i) = sum(decision_NB == 2 & labels == 2) / Nc(2);
    pE_NB(i) = pFA_NB(i) * p(1) + (1-pCD_NB(i)) * p(2);
end
[min_error_nb, min_index_nb] = min(pE_NB);
min_decision_nb = likelihood_ratios_nb >= tau_nb(min_index_nb);
min_FA_NB = pFA_NB(min_index_nb); min_CD_NB = pCD_NB(min_index_nb);

figure(2); plot(pFA_NB, pCD_NB, '-', min_FA_NB, min_CD_NB, 'o', ideal_pFA, ideal_pCD, 'g+');
set(gca,'fontname','Linux Libertine')
legend('ROC Curve', 'Empirical Min Error', 'Theoretical Min Error');
title("ROC Curve for Naive Bayes");
xlabel("False Positive Rate");
ylabel("True Positive Rate");
saveas(gcf, 'q1_b_roc.png');



% Question 1 Part C
x1 = x(:, labels==1);
x2 = x(:, labels==2);
% Estimate mean vectors and covariance matrices from samples
mu1hat = mean(x1,2); S1hat = cov(x1'); mu2hat = mean(x2,2); S2hat = cov(x2');
Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)'; Sw = S1hat + S2hat;
[V,D] = eig(inv(Sw)*Sb);
[~, ind] = sort(diag(D), 'descend');
wLDA = V(:, ind(1));
w=wLDA;
yLDA = wLDA'*x;
wLDA = sign(mean(yLDA(find(labels==2))) - mean(yLDA(find(labels==1)))) * wLDA;
yLDA = sign(mean(yLDA(find(labels==2))) - mean(yLDA(find(labels==1)))) * yLDA;
tauLDA = sort(yLDA);
for i=1:length(tauLDA)
    decisionsLDA = yLDA > tauLDA(i);
    decisionsLDA = decisionsLDA + 1;
    fprLDA(i) = sum(decisionsLDA == 2 & labels == 1) / Nc(1);
    tprLDA(i) = sum(decisionsLDA == 2 & labels == 2) / Nc(2);
    peLDA(i) = fprLDA(i) * p(1) + (1-tprLDA(i)) * p(2);
end
[min_error_LDA, min_index_LDA] = min(peLDA);
minFprLDA = fprLDA(min_index_LDA); minTprLDA = tprLDA(min_index_LDA);

figure(2); plot(fprLDA, tprLDA, '-', minFprLDA, minTprLDA, 'o', ideal_pFA, ideal_pCD, 'g+');
set(gca,'fontname','Linux Libertine')
legend('ROC Curve', 'Empirical Min Error', 'Theoretical Min Error');
title("ROC Curve for LDA");
xlabel("False Positive Rate");
ylabel("True Positive Rate");
saveas(gcf, 'q1_c_roc.png');

close all


