###Rstan
rm(list=ls())

install.packages("Rcpp", repos = "https://rcppcore.github.io/drat")
remove.packages("rstan")
if (file.exists(".RData")) file.remove(".RData")

Sys.setenv(MAKEFLAGS = "-j4") # four cores used

install.packages("rstan", type = "source")


############
library(splines)
library(numDeriv)
library(tictoc)
library(SMUT) ## %*%->eigenMapMatMult
library(Rfast) ## t->transpose

dz <- 0.02 #space step
dt <- 0.0001 #time step
t.s <- 0.1 # st deviation true value

T = 0.005 #Whole time
a= 0
b=1
t.theta1 = 0.165  # True value for theta1
t.theta2 = 0.2    # True value for theta2


tstep <- T/dt
N <- (b-a)/dz

u <- matrix(0,nr=tstep+1, nc = N+1) #numerical solution


# Boundary data; u(0,t) and u(1,t) = e^(-t)
for(k in 0:tstep+1){
  u[k,1] <- exp(-(k-1)*dt)
  u[k,N+1] <- exp(-(k-1)*dt)
}

#Initial data; u(x,0) = 1 -sin(x)
for(i in 1:N+1){
  u[1,i] <- 1-sin(pi*(i-1)*dz)
}



# Numerical solution computation for true values, matrix version
A <- matrix(0, N-1, N+1)
for(i in 1:N-1){
  A[i,i] <-(t.theta1*dt/dz/dz + t.theta2*dt/2/dz)
  A[i,i+1] <- (1-2*t.theta1*dt/dz/dz)
  A[i,i+2] <- (t.theta1*dt/dz/dz - t.theta2*dt/2/dz)
}

A <- t(A)
for(k in 1:tstep){
  u[k+1,c(2:N)] <-u[k,,drop=FALSE]%*%A
}

# Numerical solution computation for true values, matrix version
A <- matrix(0, N-1, N+1)
for(i in 1:N-1){
  A[i,i] <-(t.theta1*dt/dz/dz + t.theta2*dt/2/dz)
  A[i,i+1] <- (1-2*t.theta1*dt/dz/dz)
  A[i,i+2] <- (t.theta1*dt/dz/dz - t.theta2*dt/2/dz)
}

A <- t(A)
for(k in 1:tstep){
  u[k+1,c(2:N)] <-u[k,,drop=FALSE]%*%A
}

# random data generation
set.seed(1000)
y <-matrix(nr=tstep+1, nc = N+1)
y <- u + replicate(N+1, rnorm(tstep+1,0,t.s))
vec_y<-c(y)
n<-length(vec_y)


# Define basis matrix
df_x<-12
df_t<-12
K<-df_x*df_t
nz<-(b-a)/dz+1
nt<-(T-0)/dt+1

B1<-bs(seq(a,b,by=dz),degree=3,df=df_x,intercept = TRUE)## df-degree is # of inner knots
B2<-bs(seq(0,T,by=dt),degree=3,df=df_t,intercept = TRUE)
B<-kronecker(B1,B2,FUN="*")
t_B<-t(B)


##define derivative matrix
dz_1<-diag(1,df_x,df_x)-cbind(0,diag(1,df_x)[,1:(df_x-1)])
dz_2<-dz_1%*%dz_1

Dt_1<-diag(1,df_t,df_t)-cbind(0,diag(1,df_t)[,1:(df_t-1)])
Dt_2<-Dt_1%*%Dt_1


##
d1<-diag(1,nz,nz)-cbind(0,diag(1,nz)[,1:(nz-1)])
d2<-d1%*%d1

dB1<-d1%*%B1
dB2<-d1%*%B2
ddB1<-d2%*%B1


##Define F at stan
Bz<-kronecker(dB1,B2,FUN="*")/dz*(dz)^2
Bt<-kronecker(B1,dB2,FUN="*")/dt*(dz)^2
Bzz<-kronecker(ddB1,B2,FUN="*")/(dz)^2*(dz)^2


###Define H for prior of beta
H1<-kronecker(t(B1)%*%B1,t(Dt_2)%*%Dt_2,FUN="*")
H2<-kronecker(t(dz_2)%*%dz_2,t(B2)%*%B2,FUN="*")
H3<-kronecker(t(dz_2)%*%dz_2,t(Dt_2)%*%Dt_2,FUN="*")


####hyperparameter
a_g<-c(1,1,1)
b_g<-c(3,3,3)
a_e<-1
b_e<-0.2
a_t<-1
b_t<-0.2
sig_t<-0.01


###############Start stan
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

data<-list(N=n,K=K,Y=vec_y,B=B,Bt=Bt,Bzz=Bzz,Bz=Bz,H1=H1,H2=H2,H3=H3,
           sig_t=sig_t,a_t=a_t,b_t=b_t,a_g=a_g,b_g=b_g,a_e=a_e,b_e=b_e)
fit<-stan(file="Desktop/pm/p-spline/p-spline.stan",data=data,seed=2,iter=1000,chains=4,cores=4,algorithm = "NUTS")
options(max.print = 2000)
save(fit,file="~/Desktop/pm/p-spline/fit.RData")

############ trace plot
install.packages("ggmcmc")
library(ggmcmc)

fit_ggs<-ggs(fit, stan_include_auxiliar = TRUE) ### extract lp__ value
fit_theta<-fit_ggs%>%
  filter(Parameter=="theta1"|Parameter=="theta2")
ggs_traceplot(fit_theta) + theme_minimal()



