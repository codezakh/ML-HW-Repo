function px = evalGaussianPDF(x,mu,Sigma)
% x should have n-dimensional N vectors in columns
n = size(x,1); % data vectors have n-dimensions
N = size(x,2); % there are N vector-valued samples
C = (2*pi)^(-n/2)*det(Sigma)^(-1/2); % normalization constant
a = x-repmat(mu,1,N); b = inv(Sigma)*a;
% a,b are preparatory random variables, in an attempt to avoid a for loop
px = C*exp(-0.5*sum(a.*b,1)); % px is a row vector that contains p(x_i) values