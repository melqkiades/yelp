% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.



%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  

load moviedata

num_m = 3952;
num_p = 6040;
count = zeros(num_p,num_m,'single'); %for Netflida data, use sparse matrix instead. 

movielens_data = dlmread('/Users/fpena/UCC/Thesis/datasets/uncompressed/ml-100k/u.data');
disp(size(movielens_data,1));
num_records = size(movielens_data,1);
train_pct = 0.9;
train_size = train_pct * size(movielens_data,1);
disp(train_size);
train_vec = movielens_data(1:train_size,1:3);
probe_vec = movielens_data(train_size+1:end,1:3);
disp('Num movies');
disp(size(unique(train_vec(:,1))));
disp(size(unique(probe_vec(:,2))));
num_p = size(unique(movielens_data(:,2)));
num_m = size(unique(movielens_data(:,1)));

for mm=1:num_m
 ff= find(train_vec(:,2)==mm);
 count(train_vec(ff,1),mm) = train_vec(ff,3);
end 


