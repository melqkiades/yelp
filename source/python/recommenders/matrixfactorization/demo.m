restart=1;
% fprintf(1,'Running Probabilistic Matrix Factorization (PMF) \n');
% pmf

restart=1;
fprintf(1,'\nRunning Bayesian PMF\n');
bayespmf

% rand('state',mtstate(0));
% randn('state',mtstate(0));
% rand('twister',0);
% s = RandStream('mt19937ar','Seed',0);
% RandStream.setGlobalStream(s);
% rng(0,'twister');
% randn(1,10);
% randg('state',0);
% disp(randg(1));

% Z = repmat(0,500,1); Z(1)=3;Z(2)=1;
% Za = (Z-repmat(mean(Z),500,1)) ./ repmat(std(Z),500,1);
% disp(Za(1));

% disp(rand());
% disp(normal_random());

% disp(rand());
% l = normal_random_matrix(2,3)
% disp(l(2,:));
% rand('state',mtstate(4));
% m = rand(1,3)
% b = isequal(l,m)
% format long
% bb = 1;
% aaa1 = 1.0079376598851518;

% for i=1:10000
%   bb = aaa1 * bb;
% end
% disp(bb);

% format long

% my_rnd_matrix = normal_random_matrix(3, 500000);
% my_dot = my_rnd_matrix * my_rnd_matrix';
% disp(my_dot);
% disp(my_dot(1,1) - 499404);


