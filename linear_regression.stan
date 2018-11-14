data{
    int N; // number of data points
    real<lower=0> y[N]; // target value: number of streams
    real acousticness[N];
    real danceability[N];
    real energy[N];
    real loudness[N];
    real tempo[N];
    real valence[N];
}

parameters{
    real a;
    real b1;
    real b2;
    real b3;
    real b4;
    real b5;
    real b6;
    real<lower=0> sigma;
}

model{
    for (i in 1:N)
        y[i] ~ normal(a+b1*acousticness[i]+b2*danceability[i]+b3*energy[i]+b4*loudness[i]+b5*tempo[i]+b6*valence[i],sigma);
}