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

% pkg load statistics

%rand('state',0);
% randn('state',0);
% rand('state',mtstate(0));
% rand('twister',0);
format long

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave
        file_name = '/Users/fpena/tmp/bpmf/octave-results.txt';
        rand('state',mtstate(0));
else
        file_name = '/Users/fpena/tmp/bpmf/matlab-results.txt';
        rand('twister',0);
end

fileID = fopen(file_name,'w');

% if restart==1 
  restart=0; 
  epoch=1; 
  maxepoch=50; 

  iter=0; 
  num_features = 10;

  % Initialize hierarchical priors 
  beta=2; % observation noise (precision) 
  mu_user = zeros(num_features,1);
  mu_item = zeros(num_features,1);
  alpha_user = eye(num_features);
  alpha_item = eye(num_features);  

  % parameters of Inv-Whishart distribution (see paper for details) 
  WI_user = eye(num_features);
  b0_user = 2;
  df_user = num_features;
  mu0_user = zeros(num_features,1);

  WI_item = eye(num_features);
  b0_item = 2;
  df_item = num_features;
  mu0_item = zeros(num_features,1);

  load moviedata
  mean_rating = mean(train_vec(:,3));
  ratings_test = double(probe_vec(:,3));

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  movielens_data = dlmread('/Users/fpena/tmp/bpmf/u.data');
  % movielens_data = dlmread('/Users/fpena/tmp/bpmf/u-1k.data');
  num_records = size(movielens_data,1);
  
  % We shuffle the data
  % indexes = randperm(num_records);
  % movielens_data = movielens_data(indexes,:);

  train_pct = 0.9;
  train_size = train_pct * size(movielens_data,1);
  train_vec = movielens_data(1:train_size,1:3);
  probe_vec = movielens_data(train_size+1:end,1:3);
  mean_rating = mean(train_vec(:,3));
  ratings_test = double(probe_vec(:,3));

  % disp('mean_rating');
  % disp(mean_rating);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  pairs_tr = length(train_vec);
  pairs_pr = length(probe_vec);

  fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 
  makematrix

%   load pmf_weight
%   err_test = cell(maxepoch,1);
% 
%   user_features_l = user_features; 
%   item_features_l = item_features; 
%   clear user_features item_features;

  num_items = size(unique(movielens_data(:,2)),1);
  num_users = size(unique(movielens_data(:,1)),1);

  df_mpost_item = df_item+num_items;  % This is a constant (num_items + num_features) 
  df_mpost_user = df_user+num_users; % This is a constant (num_users + num_features)
  
  item_features_l = 0.1*normal_random_matrix(num_items, num_features); % Item feature vectors
  user_features_l = 0.1*normal_random_matrix(num_users, num_features); % User feature vectors
  



  % Initialization using MAP solution found by PMF. 
  %% Do simple fit
  % mu_user = mean(user_features_l)';
  %d=num_features;
  % alpha_user = inv(cov(user_features_l));

  % mu_item = mean(item_features_l)';
  % alpha_item = inv(cov(user_features_l));

  count=count';
  probe_rat_all = pred(item_features_l,user_features_l,probe_vec,mean_rating);
  counter_prob=1; 

% end


for epoch = epoch:maxepoch

  fprintf(fileID,'Epoch %d\n', epoch);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from movie hyperparams (see paper for details)  
  x_bar = mean(item_features_l)'; 
  S_bar = cov(item_features_l); 

  WI_post = inv(inv(WI_item) + num_items*S_bar + ...
            num_items*b0_item*(mu0_item - x_bar)*(mu0_item - x_bar)'/(b0_item+num_items));
  WI_post = (WI_post + WI_post')/2;
  alpha_item = wishart(WI_post,df_mpost_item,[]);
  mu_temp = (b0_item*mu0_item + num_items*x_bar)/(b0_item+num_items);
  lam = chol( inv((b0_item+num_items)*alpha_item) ); lam=lam';
  mu_item = lam*normal_random_matrix(num_features,1)+mu_temp;
  
  % disp('x_bar');
  % disp(size(x_bar));
  % disp(x_bar);
  % disp('S_bar');
  % disp(size(S_bar));
  % disp(S_bar);
  % disp('WI_post');
  % disp(size(WI_post));
  % disp(WI_post);
  % disp('df_mpost_item');
  % disp(size(df_mpost_item));
  % disp(df_mpost_item);
  % disp('alpha_item');
  % disp(size(alpha_item));
  % disp(alpha_item);
  % disp('mu_item');
  % disp(size(mu_item));
  % disp(mu_item);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from user hyperparams
  x_bar = mean(user_features_l)';
  S_bar = cov(user_features_l);

  WI_post = inv(inv(WI_user) + num_users*S_bar + ...
            num_users*b0_user*(mu0_user - x_bar)*(mu0_user - x_bar)'/(b0_user+num_users));
  WI_post = (WI_post + WI_post')/2;
  alpha_user = wishart(WI_post,df_mpost_user,[]);
  mu_temp = (b0_user*mu0_user + num_users*x_bar)/(b0_user+num_users);
  lam = chol( inv((b0_user+num_users)*alpha_user) ); lam=lam';
  mu_user = lam*normal_random_matrix(num_features,1)+mu_temp;

  size_alpha_item = size(alpha_item);
  size_mu_item = size(mu_item);
  size_alpha_user = size(alpha_user);
  size_mu_user = size(mu_user);
  fprintf(fileID,'alpha_item \t (%d,%d) \t %16.16f\n', size_alpha_item(1), size_alpha_item(2), alpha_item(1,1));
  fprintf(fileID,'mu_item \t (%d,%d) \t %16.16f\n', size_mu_item(1), size_mu_item(2), mu_item(1,1));
  fprintf(fileID,'alpha_user \t (%d,%d) \t %16.16f\n', size_alpha_user(1), size_alpha_user(2), alpha_user(1,1));
  fprintf(fileID,'mu_user \t (%d,%d) \t %16.16f\n', size_mu_user(1), size_mu_user(2), mu_user(1,1));

  

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Start doing Gibbs updates over user and 
  % movie feature vectors given hyperparams.  

  for gibbs=1:2 
    fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);
    fprintf(fileID,'Gibbs cycle %d\n', gibbs);

    %%% Infer posterior distribution over all movie feature vectors 
    count=count';
    % disp('Items');
    % disp(num_items);
    % disp('count');
    % disp(size(count));
    for item=1:num_items
       fprintf(1,'movie =%d\r',item);
       fprintf(fileID,'Item %d\n', item);
       users = find(count(:,item)>0);  % finds the position of the user that have rated the item
       features = user_features_l(users,:); % obtains the features of the users who have rated the item
       ratings = count(users,item)-mean_rating;  % obtains the ratings given by users for the item minus the mean rating
       % disp('count');
       % disp(class(count));
       % disp(size(count(users,item)));
       % disp(count(users,item)(1:5, :));

       % disp('Sum');
       % disp(5 - mean_rating);
       % disp('ratings00');
       % disp(size(ratings));
       % disp(ratings(1:5, :));
       covar = inv(alpha_item+beta*features'*features);  % equation 12 (without the inverse (-1)).
       mean_m = covar * (beta*features'*ratings+alpha_item*mu_item);  % equation 13, but here M (which is supposed to be V_j) is being transposed
       lam = chol(covar); lam=lam'; 
       temp_m = lam*normal_random_matrix(num_features,1)+mean_m;
       item_features_l(item,:) = temp_m;  % equation 11
       % size_alpha_item = size(alpha_item);
       size_covar = size(covar);
       size_ratings = size(ratings);
       size_features = size(features);
       % fprintf(fileID,'alpha_item \t (%d,%d) \t %16.16f\n', size_alpha_item(1), size_alpha_item(2), alpha_item(1,1));
       % fprintf(fileID,'beta \t %16.16f\n', beta);
       % if ~isempty(ratings)
       %   fprintf(fileID,'ratings \t (%d,%d) \t %16.16f\n', size_ratings(1), size_ratings(2), ratings(1, 1));
       % else
       %   fprintf(fileID,'No ratings\n');
       % end
       % if ~isempty(features)
       %   fprintf(fileID,'features \t (%d,%d) \t %16.16f\n', size_features(1), size_features(2), features(1, 1));
       % else
       %   fprintf(fileID,'No features\n');
       % end
       % fprintf(fileID,'covar \t (%d,%d) \t %16.16f\n', size_covar(1), size_covar(2), covar(1, 1));
       % Note that what here is called beta in the paper is called alpha.
       % And what here is called alpha_item in the paper is called alpha_item in the paper is called Lambda_m
       if item == 1
         % disp(size(user_features_l));
         % disp('users');
         % disp(size(users));
         % disp(users);
         % disp('features');
         % disp(size(features));
         % disp(features(1:5, :));
         % disp('alpha_item');
         % disp(size(alpha_item));
         % disp(alpha_item);
         % disp('ratings');
         % disp(size(ratings));
         % disp(ratings(1:5, :));
         % disp('covar');
         % disp(size(covar));
         % disp(covar);
         % disp('lam');
         % disp(size(lam));
         % disp(lam);
         % disp('temp_m');
         % disp(size(temp_m));
         % disp(temp_m);
         % disp('mean_m');
         % disp(size(mean_m));
         % disp(mean_m);
         % disp('user_features_l');
         % disp(size(user_features_l));
         % disp(user_features_l(1:10, 1:5));
       end

       
       % size_features = size(features);
       % fprintf(fileID,'features \t (%d,%d) \t %16.16f\n', size_features(1), size_features(2), features(1:1,:));
       % size_ratings = size(ratings);
       % fprintf(fileID,'ratings \t (%d,%d) \t %16.16f\n', size_ratings(1), size_ratings(2), ratings(1:1,:));
     end
     % disp('item_features_l');
     % disp(size(item_features_l));
     % disp(item_features_l(1:5, :));
     size_item_feaures_l = size(item_features_l);
     fprintf(fileID,'item_features \t (%d,%d) \t %16.16f\n', size_item_feaures_l(1), size_item_feaures_l(2), item_features_l(1,1));

    %%% Infer posterior distribution over all user feature vectors 
     count=count';
     % disp('count');
     % disp(size(count));
     % disp(count(1:10, 1:5));
     for user=1:num_users
       fprintf(1,'user  =%d\r',user);
       fprintf(fileID,'User %d\n', user);
       items = find(count(:,user)>0);
       features = item_features_l(items,:);
       ratings = count(items,user)-mean_rating;
       covar = inv(alpha_user+beta*features'*features);
       mean_u = covar * (beta*features'*ratings+alpha_user*mu_user);
       lam = chol(covar); lam=lam'; 
       temp_u = lam*normal_random_matrix(num_features,1)+mean_u;
       user_features_l(user,:) = temp_u;

       % if ~isempty(ratings)
       %   fprintf(fileID,'ratings \t (%d,%d) \t %16.16f\n', size_ratings(1), size_ratings(2), ratings(1, 1));
       % else
       %   fprintf(fileID,'No features\n');
       % end
       % if ~isempty(features)
       %   fprintf(fileID,'features \t (%d,%d) \t %16.16f\n', size_features(1), size_features(2), features(1, 1));
       % else
       %   fprintf(fileID,'No features\n');
       % end
       % fprintf(fileID,'covar \t (%d,%d) \t %16.16f\n', size_covar(1), size_covar(2), covar(1, 1));

       if user == 1
         % disp(size(item_features_l));
         % disp('items');
         % disp(size(items));
         % disp(items);
         % disp('features');
         % disp(size(features));
         % disp(features(1:10, 1:5));
         % disp('covar');
         % disp(size(covar));
         % disp(covar);
         % disp('lam');
         % disp(size(lam));
         % disp(lam);
         % disp('temp_u');
         % disp(size(temp_u));
         % disp(temp_u);
         % disp('user_features_l');
         % disp(size(user_features_l));
         % disp(user_features_l(1:10, 1:5));
       end
       
     end
     size_user_feaures_l = size(user_features_l);
     fprintf(fileID,'user_features \t (%d,%d) \t %16.16f\n', size_user_feaures_l(1), size_user_feaures_l(2), user_features_l(1,1));
   end 

   

   % disp('mu_item');
   % disp(mu_item);
   % disp('mu_user');
   % disp(mu_user);
   % disp(item_features_l(1:5, :));
   % disp('user_features_l');
   % disp(size(user_features_l))
   % disp(user_features_l(1:5, :));

   % disp('probe_rat_all');
   % disp(size(probe_rat_all));

   probe_rat = pred(item_features_l,user_features_l,probe_vec,mean_rating);
   probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
   counter_prob=counter_prob+1;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%% Make predictions on the validation data %%%%%%%
   temp = (ratings_test - probe_rat_all).^2;
   err = sqrt( sum(temp)/pairs_pr);

   rmse = sqrt(mean((ratings_test - probe_rat).^2));  % Root Mean Squared Error

   iter=iter+1;
   overall_err(iter)=err;

  fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \t RMSE %6.4f \n', epoch, err, rmse);

end 

fclose(fileID);
