function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

[num_rows, num_cols] = size(z);

% You need to return the following variables correctly 
g = zeros(num_rows, num_cols);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i = 1:num_rows,
	for j = 1:num_cols,
		temp = 1 + exp(-(z(i,j)));
		g(i,j) = 1/temp;
	end
end

% =============================================================

end
