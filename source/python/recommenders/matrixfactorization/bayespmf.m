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

pkg load statistics

rand('state',0);
randn('state',0);

if restart==1 
  restart=0; 
  epoch=1; 
  maxepoch=50; 

  iter=0; 
  num_features = 10;

  % Initialize hierarchical priors 
  beta=2; % observation noise (precision) 
  mu_u = zeros(num_features,1);
  mu_m = zeros(num_features,1);
  alpha_u = eye(num_features);
  alpha_m = eye(num_features);  

  % parameters of Inv-Whishart distribution (see paper for details) 
  WI_u = eye(num_features);
  b0_u = 2;
  df_u = num_features;
  mu0_u = zeros(num_features,1);

  WI_m = eye(num_features);
  b0_m = 2;
  df_m = num_features;
  mu0_m = zeros(num_features,1);

  load moviedata
  mean_rating = mean(train_vec(:,3));
  ratings_test = double(probe_vec(:,3));

  pairs_tr = length(train_vec);
  pairs_pr = length(probe_vec);

  fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 
  makematrix

  load pmf_weight
  err_test = cell(maxepoch,1);

  user_features_l = user_features; 
  item_features_l = item_features; 
  clear user_features item_features;

  num_items = size(item_features_l,1);
  num_users = size(user_features_l,1);
  disp('num_items');
  disp(num_items);
  disp('num_users');
  disp(num_users);

  df_mpost_item = df_m+num_items;  % This is a constant (num_items + num_features) 
  df_mpost_user = df_u+num_users; % This is a constant (num_users + num_features)
  
  item_features_l = 0.1*randn(num_items, num_features); % Item feature vectors
  user_features_l = 0.1*randn(num_users, num_features); % User feature vectors

  


  % Initialization using MAP solution found by PMF. 
  %% Do simple fit
  % mu_u = mean(user_features_l)';
  %d=num_features;
  % alpha_u = inv(cov(user_features_l));

  % mu_m = mean(item_features_l)';
  % alpha_m = inv(cov(user_features_l));

  count=count';
  probe_rat_all = pred(item_features_l,user_features_l,probe_vec,mean_rating);
  counter_prob=1; 

end


for epoch = epoch:maxepoch

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from movie hyperparams (see paper for details)  
  x_bar = mean(item_features_l)'; 
  S_bar = cov(item_features_l); 

  WI_post = inv(inv(WI_m) + num_items*S_bar + ...
            num_items*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(b0_m+num_items));
  WI_post = (WI_post + WI_post')/2;
  alpha_m = wishrnd(WI_post,df_mpost_item,[]);
  mu_temp = (b0_m*mu0_m + num_items*x_bar)/(b0_m+num_items);
  lam = chol( inv((b0_m+num_items)*alpha_m) ); lam=lam';
  mu_m = lam*randn(num_features,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from user hyperparams
  x_bar = mean(user_features_l)';
  S_bar = cov(user_features_l);

  WI_post = inv(inv(WI_u) + num_users*S_bar + ...
            num_users*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(b0_u+num_users));
  WI_post = (WI_post + WI_post')/2;
  alpha_u = wishrnd(WI_post,df_mpost_user,[]);
  mu_temp = (b0_u*mu0_u + num_users*x_bar)/(b0_u+num_users);
  lam = chol( inv((b0_u+num_users)*alpha_u) ); lam=lam';
  mu_u = lam*randn(num_features,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Start doing Gibbs updates over user and 
  % movie feature vectors given hyperparams.  

  for gibbs=1:2 
    fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

    %%% Infer posterior distribution over all movie feature vectors 
    count=count';
    for item=1:num_items
       fprintf(1,'movie =%d\r',item);
       users = find(count(:,item)>0);  % finds the position of the user that have rated the item
       features = user_features_l(users,:); % obtains the features of the users who have rated the item
       ratings = count(users,item)-mean_rating;  % obtains the ratings given by users for the item minus the mean rating
       covar = inv((alpha_m+beta*features'*features));  % equation 12 (without the inverse (-1)).
       mean_m = covar * (beta*features'*ratings+alpha_m*mu_m);  % equation 13, but here M (which is supposed to be V_j) is being transposed
       lam = chol(covar); lam=lam'; 
       item_features_l(item,:) = lam*randn(num_features,1)+mean_m;  % equation 11
       % Note that what here is called beta in the paper is called alpha.
       % And what here is called alpha_m in the paper is called alpha_m in the paper is called Lambda_m
     end

    %%% Infer posterior distribution over all user feature vectors 
     count=count';
     for user=1:num_users
       fprintf(1,'user  =%d\r',user);
       items = find(count(:,user)>0);
       features = item_features_l(items,:);
       ratings = count(items,user)-mean_rating;
       covar = inv((alpha_u+beta*features'*features));
       mean_u = covar * (beta*features'*ratings+alpha_u*mu_u);
       lam = chol(covar); lam=lam'; 
       user_features_l(user,:) = lam*randn(num_features,1)+mean_u;
     end
   end 

   probe_rat = pred(item_features_l,user_features_l,probe_vec,mean_rating);
   probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
   counter_prob=counter_prob+1;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%% Make predictions on the validation data %%%%%%%
   temp = (ratings_test - probe_rat_all).^2;
   err = sqrt( sum(temp)/pairs_pr);

   iter=iter+1;
   overall_err(iter)=err;

  fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);

end 


