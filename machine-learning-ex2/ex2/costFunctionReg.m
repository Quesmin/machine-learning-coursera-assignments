function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);

sum1 = 0;
sum2 = 0;
h = 0;

for i = 1:m
	h = 1/(1 + e^(-(theta')*(X(i, :))'));
	sum1+= -y(i)*log(h) - (1 - y(i))*log(1 - h);
endfor

J = sum1/m;

for j = 2:n
	sum2+= (theta(j))^2;
endfor

J = J + (lambda * sum2)/(2*m);

temp = zeros(size(theta));

for i = 1:m
	temp(1)+= (1/(1 + e^(-(theta')*(X(i, :))')) - y(i))*X(i,1);
endfor
temp(1) = temp(1)/m;

for j = 2:size(theta)
	for i = 1:m
		temp(j)+= (1/(1 + e^(-(theta')*(X(i, :))')) - y(i))*X(i,j);
	endfor
temp(j) = temp(j)/m + (lambda * theta(j))/m;
endfor

grad = temp;




% =============================================================

end
