

function rand_num = normal_random();

persistent spare;
persistent is_spare_ready;

% rand_num = rand();
% disp('rand_num');
% disp(rand_num);

if is_spare_ready
	is_spare_ready = false;
	rand_num = spare;
else
	u = rand() * 2 -1;
	v = rand() * 2 -1;
	s = u * u + v * v;

	while (s>= 1 || s==0)
		u = rand() * 2 -1;
		v = rand() * 2 -1;
		s = u * u + v * v;
	end

	mul = sqrt(-2 * log(s) /s);
	spare = v * mul;
	is_spare_ready = true;
	rand_num = u * mul;
end

