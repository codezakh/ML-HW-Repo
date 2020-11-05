N = 100;
priors = [0.6 0.4];
num_true = binornd(N, priors(2));
num_false = N - num_true;
labels = cat(1, ones(1, num_false)', ones(1, num_true)' .* 2);
gm_false = gmdistribution([5 0; 0 4], cat(3, [4 0 ; 0 2], [1 0; 0 3]));
samples_false = gm_false.random(num_false);
samples_true = mvnrnd([3 2], [2 0; 0 2], num_true);
samples = cat(1, samples_false, samples_true);

makebase = @(N, priors, gm_false) struct('N', N, 'priors', priors, 'gm_false', gm_false);

train_100 = make_dataset(makebase(100, priors, gm_false));
train_1k = make_dataset(makebase(1000, priors, gm_false));
train_10k = make_dataset(makebase(10000, priors, gm_false));
val_20k = make_dataset(makebase(20000, priors, gm_false));

% Q1 Part 1
% Ideal classifier with true knowledge of PDF on Validation Set.
pxgivenl = ones(2, val_20k.N);
pxgivenl(1, :) = gm_false.pdf(val_20k.samples);
pxgivenl(2, :) = mvnpdf(val_20k.samples, [3 2], [2 0; 0 2]);
likelihood_ratios = pxgivenl(2, :) ./ pxgivenl(1, :);
likelihood_ratios = likelihood_ratios';
tau = sort(likelihood_ratios(likelihood_ratios >= 0));

Nc = [length(find(labels==1)),length(find(labels==2))];
p = priors;
for i=1:length(tau)
    decision = likelihood_ratios >= tau(i);
    decision = decision + 1;
    pFA(i) = sum(decision == 2 & val_20k.labels == 1) / val_20k.Nc(1);
    pCD(i) = sum(decision == 2 & val_20k.labels == 2) / val_20k.Nc(2);
    pE(i) = pFA(i) * p(1) + (1-pCD(i))*p(2);
end
[min_error, min_index] = min(pE);
min_decision = likelihood_ratios >= tau(min_index);
min_FA = pFA(min_index); min_CD = pCD(min_index);

% Find theoretical minimum error
ideal_decision = (likelihood_ratios > p(1)/p(2));
ideal_decision = ideal_decision + 1;
ideal_pFA = sum(ideal_decision == 2 & val_20k.labels == 1) / val_20k.Nc(1);
ideal_pCD = sum(ideal_decision == 2 & val_20k.labels == 2) / val_20k.Nc(2);
ideal_error = ideal_pFA * p(1) + (1 - ideal_pCD) * p(2);

figure(); plot(pFA, pCD, '-', min_FA, min_CD, 'o', ideal_pFA, ideal_pCD, 'g+');
set(gca,'fontname','Linux Libertine')
legend('ROC Curve', 'Empirical Min Error', 'Theoretical Min Error');
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title("Ideal Classifier");
saveas(gcf, 'q1-1_ideal_roc.png');

% Q1 Part 2
MLEGM(train_100, ideal_pFA, ideal_pCD)
MLEGM(train_1k, ideal_pFA, ideal_pCD)
MLEGM(train_10k, ideal_pFA, ideal_pCD)
close all;

% Q1 Part 3
[train_err, val_err] = MLELogLinear(train_100, val_20k);
fprintf("LogLinear, n=%d, train_err=%f, val_err=%f\n", 100, train_err, val_err);
[train_err, val_err] = MLELogLinear(train_1k, val_20k);
fprintf("LogLinear, n=%d, train_err=%f, val_err=%f\n", 1000, train_err, val_err);
[train_err, val_err] = MLELogLinear(train_10k, val_20k);
fprintf("LogLinear, n=%d, train_err=%f, val_err=%f\n", 10000, train_err, val_err);


[train_err, val_err] = MLELogQuad(train_100, val_20k);
fprintf("LogQuadratic, n=%d, train_err=%f, val_err=%f\n", 100, train_err, val_err);
[train_err, val_err] = MLELogQuad(train_1k, val_20k);
fprintf("LogQuadratic, n=%d, train_err=%f, val_err=%f\n", 1000, train_err, val_err);
[train_err, val_err] = MLELogQuad(train_10k, val_20k);
fprintf("LogQuadratic, n=%d, train_err=%f, val_err=%f\n", 10000, train_err, val_err);

clear all
%Q2 Part 
for E=1:100
    for magnitude=2:6
        sigma = zeros(2, 2, num_components);
        partition = zeros(5, 2);
        splits = [];
        sample_count = 10^magnitude;
            val_count = 20000;
        ndim = 2;
        num_components = 15;
        for i=1:num_components
            sigma(:, :, i) = diag(randi(20, 1, 2));
        end
        mu = reshape(linspace(-10, 10, num_components * 2), num_components, ndim);
        gm = gmdistribution(mu, sigma, ones(num_components, 1) ./ num_components);
        samples = gm.random(sample_count);
        samples_validation = gm.random(val_count);
        for M=1:20
            gm_est = fitgmdist(samples, M, 'CovarianceType', 'diagonal', 'RegularizationValue',0.001);
            bic(M) = (6 * M) * log(val_count) - 2 * log(sum(gm_est.pdf(samples_validation)));
        end
        [~, chosenModelOrder] = min(bic)


        %Q2 Part 2
        K=5;
        % Generate the partitions.
        partition_start = 1;
        for i=1:5
            partition_size = sample_count / K;
            partition(i, :) = [partition_start partition_start + partition_size - 1];
            partition_start = partition_start + partition_size;
        end


        for i=1:5
            mask = ones(1, 5);
            mask(i) = 0;
            splits(i).validation = partition(ones(1, 5) & mask, :);
            splits(i).training = partition(i, :);
        end

        parfor M=1:20
            likelihoods_per_split = [0 0 0 0 0];
            for i=1:5
                split = splits(i);
                validation_samples = samples(split.training(1):split.training(2), :);
                training_mask = logical(ones(sample_count, 1));
                training_mask(split.training(1):split.training(2)) = 0;
                training_samples = samples(training_mask, :);
                gm_est = fitgmdist(training_samples, M, 'CovarianceType', 'diagonal', 'RegularizationValue',0.001);
                log_likelihood = sum(log(gm_est.pdf(validation_samples)))*4/5*sample_count;
                likelihoods_per_split(i) = log_likelihood;
            end
            likelihoods_per_M(M) = mean(likelihoods_per_split);
        end
        [~, bestC] = max(likelihoods_per_M);
        bestCforMagnitude(magnitude) = bestC;
        bestCforMagnitudeBIC(magnitude) = chosenModelOrder;
        fprintf("Done with %d\n", magnitude);
    end
    kfoldcvresults(:, E) = bestCforMagnitude;
    BICresults(:, E) = bestCforMagnitudeBIC;
end

save("kfold.mat", "kfoldcvresults");
save("bic.mat", 'BICresults'); 


function ds = make_dataset(ds)
    N = ds.N;
    priors = ds.priors;
    num_true = binornd(N, priors(2));
    num_false = N - num_true;
    labels = cat(1, ones(1, num_false)', ones(1, num_true)' .* 2);
    gm_false = ds.gm_false;
    samples_false = gm_false.random(num_false);
    samples_true = mvnrnd([3 2], [2 0; 0 2], num_true);
    samples = cat(1, samples_false, samples_true);
  
    ds.samples = samples;
    ds.samples_true = samples_true;
    ds.samples_false = samples_false;
    ds.labels = labels;
    ds.num_true = num_true;
    ds.num_false = num_false;
    ds.Nc = [length(find(labels==1)),length(find(labels==2))];
end

function min_error = MLEGM(ds, ideal_pFA, ideal_pCD)
    samples_true = ds.samples_true;
    samples_false = ds.samples_false;
    num_true = ds.num_true;
    N = ds.N;
    samples = ds.samples;
    Nc = ds.Nc;
    labels = ds.labels;

    mle_true.mu = mean(samples_true);
    sigma_tensor = zeros(2,2, num_true);
    for i=1:num_true
        sigma_tensor(:, :, i) = (samples_true(i, :) - mle_true.mu)' * (samples_true(i, :) - mle_true.mu);
    end
    mle_true.sigma = mean(sigma_tensor, 3);

    % MLE parameter estimate using optimization
    mle_false = fitgmdist(samples_false, 2);
    pxgivenl = ones(2, N);
    pxgivenl(1, :) = mle_false.pdf(samples);
    pxgivenl(2, :) = mvnpdf(samples, mle_true.mu, mle_true.sigma);
    likelihood_ratios = pxgivenl(2, :) ./ pxgivenl(1, :);
    likelihood_ratios = likelihood_ratios';
    tau = sort(likelihood_ratios(likelihood_ratios >= 0));

    p = Nc / sum(Nc); % estimate the class priors
    for i=1:length(tau)
        decision = likelihood_ratios >= tau(i);
        decision = decision + 1;
        pFA(i) = sum(decision == 2 & labels == 1) / Nc(1);
        pCD(i) = sum(decision == 2 & labels == 2) / Nc(2);
        pE(i) = pFA(i) * p(1) + (1-pCD(i))*p(2);
    end
    [min_error, min_index] = min(pE);
    min_FA = pFA(min_index); min_CD = pCD(min_index);
    figure(); plot(pFA, pCD, '-', min_FA, min_CD, 'o', ideal_pFA, ideal_pCD, 'g+');
    set(gca,'fontname','Linux Libertine')
    legend('ROC Curve', 'Empirical Min Error', 'Theoretical Min Error');
    xlabel("False Positive Rate");
    ylabel("True Positive Rate");
    title(sprintf("MLE for GM with n=%d samples", N));
    saveas(gcf, sprintf('q1-2_%d_roc.png', N));
end

function [train_err, val_err] = MLELogLinear(ds, val_ds)
    samples_true = ds.samples_true;
    samples_false = ds.samples_false;
    num_true = ds.num_true;
    N = ds.N;
    samples = ds.samples;
    Nc = ds.Nc;
    labels = ds.labels;
    p = Nc / sum(Nc);
    
    
    h = @(x, w) 1 ./ (1 + exp(-x * w'));
    theta = @(x) cat(2, ones(size(x,1), 1), x);
    thetaquad = @(x) cat(2, ones(size(x, 1), 1), [x(:, 1) x(:, 2) x(:, 1).^2 x(:,1).*x(:,2) x(:,2).^2]);
    w_linear = randn(size(theta(samples_true(1, :))));
    w_quad = randn(size(thetaquad(samples_true(1, :))));
    crs_entropy_1 = @(x, w, labels) sum(labels' * log(h(x, w)));
    crs_entropy_2 = @(x, w, labels) sum((1 - labels)' * log(1-h(x,w)));
    cost = @(x, w, labels, N) (-1/N) * (crs_entropy_1(x,w,labels) + crs_entropy_2(x, w, labels));
    cost(theta(samples), w_linear, labels - 1, size(labels, 1));
    
    [w_linear_est, cost_L] = fminsearch(@(w) cost(theta(samples), w, labels-1, size(labels, 1)), w_linear);
    discriminant_scores = theta(samples) * w_linear_est';
    decision = discriminant_scores >= 1;
    decision = decision + 1;
    pFA = sum(decision == 2 & labels == 1) / Nc(1);
    pCD = sum(decision == 2 & labels == 2) / Nc(2);
    pE = pFA * p(1) + (1-pCD)*p(2)
    train_err = pE;


 
    discriminant_scores = theta(val_ds.samples) * w_linear_est';
    decision = discriminant_scores >= 1;
    decision = decision + 1;
    pFA = sum(decision == 2 & val_ds.labels == 1) / val_ds.Nc(1);
    pCD = sum(decision == 2 & val_ds.labels == 2) / val_ds.Nc(2);
    pE = pFA * p(1) + (1-pCD)*p(2);
    val_err = pE;
end

function [train_err, val_err] = MLELogQuad(ds, val_ds)
    samples_true = ds.samples_true;
    samples_false = ds.samples_false;
    num_true = ds.num_true;
    N = ds.N;
    samples = ds.samples;
    Nc = ds.Nc;
    labels = ds.labels;
    p = Nc / sum(Nc);
    
    
    h = @(x, w) 1 ./ (1 + exp(-x * w'));
    theta = @(x) cat(2, ones(size(x,1), 1), x);
    thetaquad = @(x) cat(2, ones(size(x, 1), 1), [x(:, 1) x(:, 2) x(:, 1).^2 x(:,1).*x(:,2) x(:,2).^2]);
    w_linear = randn(size(theta(samples_true(1, :))));
    w_quad = randn(size(thetaquad(samples_true(1, :))));
    crs_entropy_1 = @(x, w, labels) sum(labels' * log(h(x, w)));
    crs_entropy_2 = @(x, w, labels) sum((1 - labels)' * log(1-h(x,w)));
    cost = @(x, w, labels, N) (-1/N) * (crs_entropy_1(x,w,labels) + crs_entropy_2(x, w, labels));
    cost(theta(samples), w_linear, labels - 1, size(labels, 1));


    [w_quad_est, cost_L] = fminsearch(@(w) cost(thetaquad(samples), w, labels-1, size(labels, 1)), w_quad);
    discriminant_scores = thetaquad(samples) * w_quad_est';
    decision = discriminant_scores >= 1;
    decision = decision + 1;
    pFA = sum(decision == 2 & labels == 1) / Nc(1);
    pCD = sum(decision == 2 & labels == 2) / Nc(2);
    pE = pFA * p(1) + (1-pCD)*p(2);
    train_err = pE;

 
    discriminant_scores = thetaquad(val_ds.samples) * w_quad_est';
    decision = discriminant_scores >= 1;
    decision = decision + 1;
    pFA = sum(decision == 2 & val_ds.labels == 1) / val_ds.Nc(1);
    pCD = sum(decision == 2 & val_ds.labels == 2) / val_ds.Nc(2);
    pE = pFA * p(1) + (1-pCD)*p(2);
    val_err = pE;
end


