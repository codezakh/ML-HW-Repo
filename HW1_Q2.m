% Question 2%
clear all, close all
C = 4;
N = 10000;
dim = 3;
% Question 2
square_corners = [
    [1 1 1]
    [-1 -1 -1]
    [-1 1 -1]
    [1 -1 1]
    ];
gmmParameters.meanVectors = square_corners';
% Distance betwen the means is 2.8284, so 
% we set the covariance matrices to have 20% of that value.
Sigma = eye(3) * 2.8284 * 0.2;
gmmParameters.covMatrices = ones(3, 3, 4);
for i=1:C
    gmmParameters.covMatrices(:, :, i) = Sigma;
end
gmmParameters.priors = [0.2, 0.25, 0.25, 0.3];

[x, labels] = generateDataFromGMM(N, gmmParameters);


for l = 1:C
    Nclass(l,1) = length(find(labels==l));
end

for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF(x,gmmParameters.meanVectors(:,l),gmmParameters.covMatrices(:,:,l)); % Evaluate p(x|L=l)
end

px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1); % P(L=l|x)

lossMatrixA = ones(C,C)-eye(C); % For min-Perror design, use 0-1 loss
expectedRisksA = lossMatrixA*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisionsA] = min(expectedRisksA,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

shapes = ['.', 'o', 's', '^'];
hold on
for i = 1:C
    mask_true = labels == i & decisionsA == i;
    mask_false = labels == i & decisionsA ~= i;
    truestyle = strcat(shapes(i), 'g');
    falsestyle = strcat(shapes(i), 'r');
    disp(truestyle)
    plot3(x(1, mask_true), x(2, mask_true), x(3, mask_true), truestyle);
    plot3(x(1, mask_false), x(2, mask_false), x(3, mask_false), falsestyle);
    set(gca,'fontname','Linux Libertine')
    title("Classification Results for 0-1 Loss")
end
hold off

for d = 1:C % each decision option
    for l = 1:C % each class label
        ind_dl = find(decisionsA==d & labels==l);
        ConfusionMatrix(d,l) = sum(ind_dl)/sum(find(labels==l));
    end
end
ConfusionMatrix,



% Question 2 Part B %
lossMatrixA = [
    [0 1 2 3]
    [10 0 5 10]
    [20 10 0 1]
    [30 20 1 0]
    ]
expectedRisksA = lossMatrixA*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisionsA] = min(expectedRisksA,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP
for i=1:length(decisionsA)
    predicted = decisionsA(i); 
    true_label = labels(i);
    incurred_risk(i) = lossMatrixA(predicted, true_label);
end
hold on
for i = 1:C
    mask_true = labels == i & decisionsA == i;
    mask_false = labels == i & decisionsA ~= i;
    truestyle = strcat(shapes(i), 'g');
    falsestyle = strcat(shapes(i), 'r');
    disp(truestyle)
    plot3(x(1, mask_true), x(2, mask_true), x(3, mask_true), truestyle);
    plot3(x(1, mask_false), x(2, mask_false), x(3, mask_false), falsestyle);
    set(gca,'fontname','Linux Libertine')
    title("Classification Results for Custom Loss");
end
hold off

estimated_risk = sum(incurred_risk) / length(incurred_risk)
for d = 1:C % each decision option
    for l = 1:C % each class label
        ind_dl = find(decisionsA==d & labels==l);
        ConfusionMatrix(d,l) = sum(ind_dl)/sum(find(labels==l));
    end
end
ConfusionMatrix,
