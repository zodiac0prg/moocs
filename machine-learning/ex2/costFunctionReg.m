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

prediction = X * theta;
h = sigmoid(prediction);
first_term = - (y' * log(h));
second_term = (1-y') * log(1-h);
cost_term = sum(first_term - second_term)/m;

# Since we should not be regularizing for theta(1)
theta_alt = theta;
theta_alt(1) = 0;

reg_term = sum(theta_alt .^ 2) * (lambda/(2 * m));

J = cost_term + reg_term;

diff_term = sigmoid(X * theta) - y;
inner = X' * diff_term;
grad_cost_term = inner / m;
grad_reg_term = theta_alt * (lambda/m);

grad = grad_cost_term + grad_reg_term;


% =============================================================

end
