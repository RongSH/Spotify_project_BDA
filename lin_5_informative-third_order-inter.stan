data{
    // year 2017 top 100 songs
    int N;        // number of data points
    vector[N] y;  // streams
    vector[N] X1; // acousticness
    vector[N] X2; // danceability
    vector[N] X3; // loudness
    vector[N] X4; // tempo
    vector[N] X5; // valence

    // year 2018 top 100 songs
    int Npred;        
    vector[Npred] X1pred; 
    vector[Npred] X2pred;
    vector[Npred] X3pred;
    vector[Npred] X4pred;
    vector[Npred] X5pred;
    
    // prior means 
    real mu_a;
    vector[19] mu_b;
   
    // prior stds
    real sigma_0;
}

parameters{
    real a;
    vector[19] b;
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
    vector[N] X111;
    vector[N] X222;
    vector[N] X333;
    vector[N] X444;
    vector[N] X555;
    
    X1X5 = X1 .* X5;
    X2X5 = X2 .* X5;
    X3X5 = X3 .* X5;
    X2X4 = X2 .* X4;
    
    X11 = X1 .* X1;
    X22 = X2 .* X2;
    X33 = X3 .* X3;
    X44 = X4 .* X4;
    X55 = X5 .* X5;
    
    X111 = X1 .* X11;
    X222 = X2 .* X22;
    X333 = X3 .* X33;
    X444 = X4 .* X44;
    X555 = X5 .* X55;
    
    mu = a + b[1]*X1 + b[2]*X2 + b[3]*X3 + b[4]*X4 + b[5]*X5 + b[6]*X11 + b[7]*X22 + b[8]*X33 + b[9]*X44 + b[10]*X55 + b[11]*X111 + b[12]*X222 + b[13]*X333 + b[14]*X444 + b[15]*X555 + b[16]*X2X4 + b[17]*X3X5 + b[18]*X1X5 + b[19]*X2X5;
}

model{
    a ~ normal(mu_a, sigma_0);
    b ~ normal(mu_b, sigma_0);
    y ~ normal(mu, sigma);
}

generated quantities{
    vector[Npred] ypred;
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
    
    vector[Npred] X111pred;
    vector[Npred] X222pred;
    vector[Npred] X333pred;
    vector[Npred] X444pred;
    vector[Npred] X555pred;
    
    X11pred = X1pred .* X1pred;
    X22pred = X2pred .* X2pred;
    X33pred = X3pred .* X3pred;
    X44pred = X4pred .* X4pred;
    X55pred = X5pred .* X5pred;
    
    X111pred = X1pred .* X11pred;
    X222pred = X2pred .* X22pred;
    X333pred = X3pred .* X33pred;
    X444pred = X4pred .* X44pred;
    X555pred = X5pred .* X55pred;
    
    X2X4pred = X2pred .* X4pred;
    X1X5pred = X1pred .* X5pred;
    X2X5pred = X2pred .* X5pred;
    X3X5pred = X3pred .* X5pred;
    
    
    for (j in 1:N){ 
        // logarithmic likelihood for streams (PSIS-LOO)
        log_lik[j] = normal_lpdf(y[j] | mu, sigma);
    }
    
    for (j in 1:Npred){
        // prediction for year 2018
        ypred[j] = normal_rng(a + b[1]*X1pred[j] + b[2]*X2pred[j] + b[3]*X3pred[j] + b[4]*X4pred[j] + b[5]*X5pred[j] + b[6]*X11pred[j] + b[7]*X22pred[j] + b[8]*X33pred[j] + b[9]*X44pred[j] + b[10]*X55pred[j] + b[11]*X111pred[j] + b[12]*X222pred[j] + b[13]*X333pred[j] + b[14]*X444pred[j] + b[15]*X555pred[j] + b[16]*X2X4pred[j] + b[17]*X3X5pred[j] + b[18]*X1X5pred[j] + b[19]*X2X5pred[j], sigma);
    } 
 
}