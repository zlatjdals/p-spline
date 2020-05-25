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
  matrix zeta_norm(matrix Bt,matrix Bz,matrix Bzz,vector beta,real theta){
    vector non_linear_term[N];
    for(i in 1:N){
      non_linear_term[i]=B[i]*beta*beta'*Bz[i]';
    }
    return (Bt*beta+non_linear_term-theta*Bzz*beta)'*(Bt*beta+non_linear_term-theta*Bzz*beta);
  }
}

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; //number of data
  int<lower=0> K; //number of b-splines basis
  vector[N] Y; //data
  matrix[N,K] B;
  matrix[N,K] Bt;
  matrix[N,K] Bz;
  matrix[N,K] Bzz;
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
  real<lower=0> theta;
  vector[K] beta;
}


model{
  target+=normal_lpdf(Y|B*beta,sqrt(sigep));//Y
  target+=K*log(prod(gamma))/2-beta'*(gamma[2]*H1+gamma[3]*H2+gamma[2]*gamma[3]*H3)*beta/2-gamma[1]*zeta_norm/2;
  target+=inv_gamma_lpdf(theta|a_t,b_t);//theta1
  target+=gamma_lpdf(gamma[1]|a_g,b_g);//gamma
  target+=inv_gamma_lpdf(sigep|a_e,b_e);//sigep
  
}
