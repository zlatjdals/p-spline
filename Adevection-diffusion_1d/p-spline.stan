//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

functions { 
  //  real priorbeta_lpdf(vector beta,real theta1,real theta2,vector gamma,real K,matrix Bt,matrix Bz,matrix Bzz,matrix H1,matrix H2,matrix H3){
  //   return (K/2)*log(gamma[1]*gamma[2]*gamma[3])-beta'*(gamma[1]*(Bt-theta1*Bzz+theta2*Bz)'*(Bt-theta1*Bzz+theta2*Bz)+gamma[2]*H1+gamma[3]*H2+gamma[2]*gamma[3]*H3)*beta/2;
  matrix F(real theta1,real theta2,matrix Bt,matrix Bz,matrix Bzz){
    return Bt-theta1*Bzz+theta2*Bz;
  }
}

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; //number of data
  int<lower=0> K; //number of b-splines basis
  vector[N] Y; //data
  matrix[N,K] B;
  matrix[N,K] Bt;
  matrix[N,K] Bzz;
  matrix[N,K] Bz;
  matrix[K,K] H1;
  matrix[K,K] H2;
  matrix[K,K] H3;
   
  //hyperparameter
  real<lower=0> sig_t;
  real<lower=0> a_t;
  real<lower=0> b_t;
  real<lower=0> a_e;
  real<lower=0> b_e;
  vector<lower=0>[3] a_g;
  vector<lower=0>[3] b_g;
}


parameters {
  vector<lower=0>[3] gamma;
  real<lower=0> sigep;
  real<lower=0> theta1;
  real theta2;
  vector[K] beta;
}



// model {
//   for(i in 1:N){
//     Y[i] ~ normal((B*beta)[i],sigep);
//   }
//   beta ~ ;
//   theta2 ~ normal(0,sig_t);
//   theta1 ~ inv_gamma(a_t,b_t);
//   for(i in 1:3){
//     gamma[i] ~ gamma(a_g[i],b_g[i]);
//   }
//   sigep~inv_gamma(a_e,b_e);
// }


model{
  target+=normal_lpdf(Y|B*beta,sqrt(sigep));//Y
  target+=K*log(prod(gamma))/2-beta'*(gamma[1]*(F(theta1,theta2,Bt,Bz,Bzz))'*F(theta1,theta2,Bt,Bz,Bzz)+gamma[2]*H1+gamma[3]*H2+gamma[2]*gamma[3]*H3)*beta/2;
  target+=normal_lpdf(theta2|0,sqrt(sig_t));// theta2
  target+=inv_gamma_lpdf(theta1|a_t,b_t);//theta1
  // for(i in 1:3){
  //   target+=gamma_lpdf(gamma[i]|a_g[i],b_g[i]); //gamma
  // }
  target+=gamma_lpdf(gamma|a_g,b_g); 
  target+=inv_gamma_lpdf(sigep|a_e,b_e);//sigep
  
}



