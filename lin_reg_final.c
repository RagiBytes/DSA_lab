#include<stdio.h>
#include<stdlib.h>
#include<math.h>
# define SIZE (int)1e5

double f(double*a,double*b,double*x){
    return *a**x-*b;
}
double loss(double*a,double*b,double*arr,int*size){
    double l=0;
    int i=0;
    while(*size>i){
        l+=(f(a,b,arr+2*i)-arr[2*i+1])*(f(a,b,arr+2*i)-arr[2*i+1])/ *size;
        i++;
    }
    return l;
}
double bgradient(double*a,double*b,double*arr,int*size,double*lr){
    double b_grad=0;
    int i=0;
    while(*size>i){
        b_grad+=(-2)*(f(a,b,arr+2*i)-arr[2*i+1])/ *size;
        i++;
    }
    return b_grad;
}

double agradient(double*a,double*b,double*arr,int*size,double*lr){
    double a_grad=0;
    int i=0;
    while(*size>i){
        a_grad+=2*(f(a,b,arr+2*i)-arr[2*i+1])*arr[2*i]/ *size;
        i++;
    }
    return a_grad;
}


void gradient_descent(double*a,double*b,double *lr,double*arr,int*size,double*neta){
    double a_grad,b_grad;
    a_grad=agradient(a,b,arr,size,lr);
    b_grad=bgradient(a,b,arr,size,lr);
    *neta=sqrt((*neta)*(*neta)+(a_grad)*(a_grad)+(b_grad)*(b_grad));
    *a-=*lr*a_grad/ *neta;
    *b-=*lr*b_grad/ *neta;
    //printf("%lf,%lf\n",*a,*b);
}


void main(){
    double x,y,l;
    int size=0;
    double arr[2*SIZE];
    while(scanf("%lf,%lf",&x,&y)==2&&SIZE>size){
        arr[2*size]=x;
        arr[2*size+1]=y;
        size++;
    }

    double a=0,b=0;

    int iter=100;
    double lr=0.1;
    double neta=0;
    while(iter--){
        gradient_descent(&a,&b,&lr,arr,&size,&neta);
        l=loss(&a,&b,arr,&size);
        //printf("HI\n");
    }
    printf("slope is %lf and y-intercept is %lf for lr %lf, iter %d and has loss %lf",a,b,lr,(int)1e4,l);
}