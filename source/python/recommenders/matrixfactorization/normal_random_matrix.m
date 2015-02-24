
function rand_matrix = normal_random_matrix(rows, columns);

matrix = zeros(rows, columns);

for row = 1:rows
	for column = 1:columns
		matrix(row, column) = normal_random();
	end
end
rand_matrix = matrix;
