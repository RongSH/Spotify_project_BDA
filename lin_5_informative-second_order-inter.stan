data{
    // year 2017 songs
    int N;        // number of data points
    vector[N] y;  // streams
    vector[N] X1; // acousticness
    vector[N] X2; // danceability
    vector[N] X3; // loudness
    vector[N] X4; // tempo
    vector[N] X5; // valence

    // year 2018 songs
    int Npred;        
    vector[Npred] X1pred; 
    vector[Npred] X2pred;
    vector[Npred] X3pred;
    vector[Npred] X4pred;
    vector[Npred] X5pred;
    
    // prior means 
    real mu_a;
    vector[14] mu_b;
   
    // prior stds
    real sigma_0;
}

parameters{
    real a;
    vector[14] b;
    real<lower=0> sigma;
}

transformed parameters{
    vector[N] mu;
    vector[N] X2X4;
    vector[N] X1X5;
    vector[N] X2X5;
    vector[N] X3X5;
    vector[N] X11;
    vector[N] X22;
    vector[N] X33;
    vector[N] X44;
    vector[N] X55;
    
    X1X5 = X1 .* X5;
    X2X5 = X2 .* X5;
    X3X5 = X3 .* X5;
    X2X4 = X2 .* X4;
    
    X11 = X1 .* X1;
    X22 = X2 .* X2;
    X33 = X3 .* X3;
    X44 = X4 .* X4;
    X55 = X5 .* X5;
    
    mu = a + b[1]*X1 + b[2]*X2 + b[3]*X3 + b[4]*X4 + b[5]*X5 + b[6]*X11 + b[7]*X22 + b[8]*X33 + b[9]*X44 + b[10]*X55 + b[11]*X2X4 + b[12]*X3X5 + b[13]*X1X5 + b[14]*X2X5;
}

model{
    sigma ~ normal(0,0.1);
    a ~ normal(mu_a, sigma_0);
    b ~ normal(mu_b, sigma_0);
    y ~ normal(mu, sigma);
}

generated quantities{
    vector[Npred] ypred_2018;
    vector[N] ypred;
    vector[N] log_lik;
    vector[Npred] X2X4pred;
    vector[Npred] X1X5pred;
    vector[Npred] X2X5pred;
    vector[Npred] X3X5pred;
    
    vector[Npred] X11pred;
    vector[Npred] X22pred;
    vector[Npred] X33pred;
    vector[Npred] X44pred;
    vector[Npred] X55pred;
    
    X11pred = X1pred .* X1pred;
    X22pred = X2pred .* X2pred;
    X33pred = X3pred .* X3pred;
    X44pred = X4pred .* X4pred;
    X55pred = X5pred .* X5pred;
    
    X2X4pred = X2pred .* X4pred;
    X1X5pred = X1pred .* X5pred;
    X2X5pred = X2pred .* X5pred;
    X3X5pred = X3pred .* X5pred;
    
    
    for (j in 1:N){ 
        // logarithmic likelihood for streams (PSIS-LOO)
        log_lik[j] = normal_lpdf(y[j] | mu[j], sigma);
        ypred[j] = normal_rng(mu[j], sigma);
    }
    
    for (j in 1:Npred){
        // prediction for year 2018
        ypred_2018[j] = normal_rng(a + b[1]*X1pred[j] + b[2]*X2pred[j] + b[3]*X3pred[j] + b[4]*X4pred[j] + b[5]*X5pred[j] + b[6]*X11pred[j] + b[7]*X22pred[j] + b[8]*X33pred[j] + b[9]*X44pred[j] + b[10]*X55pred[j] + b[11]*X2X4pred[j] + b[12]*X3X5pred[j] + b[13]*X1X5pred[j] + b[14]*X2X5pred[j], sigma);
        
    } 
 
}