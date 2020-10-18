function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

sum = 0;
h = 0;
for i=1:m
	h = 1/(1 + e^(-(theta')*(X(i, :))'));
	sum+= -y(i)*log(h) - (1 - y(i))*log(1 - h);
endfor

J = sum / m;


temp = zeros(size(theta));
for j=1:size(theta)
	for i=1:m
		temp(j)+= (1/(1 + e^(-(theta')*(X(i, :))')) - y(i))*X(i,j);
	endfor
	temp(j) = temp(j)/m;
endfor

grad = temp;






% =============================================================

end
