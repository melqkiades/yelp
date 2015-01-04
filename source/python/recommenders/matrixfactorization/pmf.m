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

rand('state',0); 
randn('state',0); 

if restart==1 
  restart=0;
  epsilon=50; % Learning rate 
  lambda  = 0.01; % Regularization parameter 
  momentum=0.8; 

  epoch=1; 
  maxepoch=5; 

  load moviedata % Triplets: {user_id, movie_id, rating} 
  mean_rating = mean(train_vec(:,3)); 
 
  num_training_records = length(train_vec); % training data 
  num_validation_records = length(probe_vec); % validation data 

  num_batches= 9; % Number of batches  
  num_items = 3952;  % Number of items 
  num_users = 6040;  % Number of users 
  num_features = 10; % Rank 10 decomposition 
  batch_size = num_training_records/num_batches; % number training triplets per batch 
  alpha = epsilon/batch_size

  item_features     = 0.1*randn(num_items, num_features); % Item feature vectors
  user_features     = 0.1*randn(num_users, num_features); % User feature vectors
  item_features_inc = zeros(num_items, num_features);
  user_features_inc = zeros(num_users, num_features);


end


for epoch = epoch:maxepoch

  % In each cycle the training vector is shuffled
  training_indexes = randperm(num_training_records);
  train_vec = train_vec(training_indexes,:);
  clear training_indexes 

  for batch = 1:num_batches
    fprintf(1,'epoch %d batch %d \r',epoch,batch);

    users   = double(train_vec((batch-1)*batch_size+1:batch*batch_size,1));
    items   = double(train_vec((batch-1)*batch_size+1:batch*batch_size,2));
    ratings = double(train_vec((batch-1)*batch_size+1:batch*batch_size,3));

    ratings = ratings-mean_rating; % Default prediction is the mean rating. 

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    predicted_rating = sum(item_features(items,:).*user_features(users,:),2);
    %f = sum( (predicted_rating - ratings).^2 + ...
    %    0.5*lambda*( sum( (item_features(items,:).^2 + user_features(users,:).^2),2)));
    % Note that in the above cost function the cost is calculated using (predicted_rating - ratings).^2
    % That means that the derivate is 2 * (predicted_rating - ratings)
    % Normally the error is divided by 2 (multiplied by 0.5) in the cost function

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    error_matrix = repmat(2*(predicted_rating - ratings),1,num_features);
    item_gradient=error_matrix.*user_features(users,:) + lambda*item_features(items,:);
    user_gradient=error_matrix.*item_features(items,:) + lambda*user_features(users,:);
    % In the above line the gradient is calculated for every rating, but it has to be
    % grouped (by summing it) for each user and item features

    d_item_features = zeros(num_items,num_features);
    d_user_features = zeros(num_users,num_features);

    % Here is where the error is grouped for each user and item
    for ii=1:batch_size
      d_item_features(items(ii),:) =  d_item_features(items(ii),:) +  item_gradient(ii,:);
      d_user_features(users(ii),:) =  d_user_features(users(ii),:) +  user_gradient(ii,:);
    end

    %%%% Update item and user features %%%%%%%%%%%
    % The update is done using the momentum technique of gradient descent

    item_features_inc = momentum*item_features_inc + alpha*d_item_features;
    item_features =  item_features - item_features_inc;

    user_features_inc = momentum*user_features_inc + alpha*d_user_features;
    user_features =  user_features - user_features_inc;
  end 

  %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
  predicted_rating = sum(item_features(items,:).*user_features(users,:),2);
  f_s = sum( (predicted_rating - ratings).^2 + ...
        0.5*lambda*( sum( (item_features(items,:).^2 + user_features(users,:).^2),2)));
  err_train(epoch) = sqrt(f_s/batch_size);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  % NN=num_validation_records;

  users = double(probe_vec(:,1));
  items = double(probe_vec(:,2));
  ratings = double(probe_vec(:,3));

  predicted_rating = sum(item_features(items,:).*user_features(users,:),2) + mean_rating;
  ff = find(predicted_rating>5); predicted_rating(ff)=5; % Clip predictions 
  ff = find(predicted_rating<1); predicted_rating(ff)=1;

  err_valid(epoch) = sqrt(sum((predicted_rating- ratings).^2)/num_validation_records);
  fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
              epoch, batch, err_train(epoch), err_valid(epoch));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %if (rem(epoch,10))==0
     %save pmf_weight item_features user_features
  %end

save pmf_weight item_features user_features

end 



