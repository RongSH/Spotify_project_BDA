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
    vector[7] mu_b;
   
    // prior stds
    real sigma_0;
}

parameters{
    real a;
    vector[7] b;
    real<lower=0> sigma;
}

transformed parameters{
    vector[N] mu;
    vector[N] X2X4;
    vector[N] X3X5;
    
    X2X4 = X2 .* X4;
    X3X5 = X3 .* X5;
    
    mu = a + b[1]*X1 + b[2]*X2 + b[3]*X3 + b[4]*X4 + b[5]*X5 + b[6]*X2X4 + b[7]*X3X5;
}

model{
    a ~ normal(mu_a, sigma_0);
    b ~ normal(mu_b, sigma_0);
    y ~ normal(mu, sigma);
}

generated quantities{
    vector[Npred] ypred_2018;
    vector[N] ypred;
    vector[N] log_lik;
    vector[Npred] X2X4pred;
    vector[Npred] X3X5pred;
    
    X2X4pred = X2pred .* X4pred; 
    X3X5pred = X3pred .* X5pred;
    
    for (j in 1:N){ 
        // logarithmic likelihood for streams (PSIS-LOO)
        log_lik[j] = normal_lpdf(y[j] | mu, sigma);
    }
    
    for (j in 1:Npred){
        // prediction for year 2018
        ypred_2018[j] = normal_rng(a + b[1]*X1pred[j] + b[2]*X2pred[j] + b[3]*X3pred[j] + b[4]*X4pred[j] + b[5]*X5pred[j] + b[6]*X2X4pred[j] + b[7]*X3X5pred[j], sigma);
        ypred[j] = normal_rng(a + b[1]*X1[j] + b[2]*X2[j] + b[3]*X3[j] + b[4]*X4[j] + b[5]*X5[j] + b[6]*X2X4[j] + b[7]*X3X5[j], sigma);

    } 
 
}