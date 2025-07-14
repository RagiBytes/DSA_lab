#include<stdio.h>
#include<stdlib.h>
#define SIZE (int)1e4

double f(double m, double c, double x){
    //printf("%lf\n",m*x+c);
    return m*x+c;
}

double m_gradient(double m,double c,int size,double a[2*size],double lr){
    double m_gradient=0;
    for(int i=0;i<size;i++){
        m_gradient+=2*lr*(f(m,c,a[2*i])-a[2*i+1])*a[2*i]/(size);
        //printf("%lf,%lf\n",a[2*i],a[2*i+1]);
    }
    //printf("%d\n",size);
    return m_gradient;
}

double c_gradient(double m,double c,int size,double a[2*size],double lr){
    double c_gradient=0;
    for(int i=0;i<size;i++){
        c_gradient+=2*lr*(f(m,c,a[2*i])-a[2*i+1])/(size);
    }
    //printf("%d\n",size);
    return c_gradient;
}

double* gradient_descent(double m,double c,double lr,int size,double a[2*size],double send[2]){
    send[0]=m-m_gradient(m,c,size,a,lr);
    send[1]=c-c_gradient(m,c,size,a,lr);
    //printf("%lf,%lf\n",send[0],send[1]);
    return send;
}



int main(){
    int size=0;
    double x,y;
    double a[2*SIZE];
    //printf("%d",scanf(" %lf %lf",&x,&y));
    while(scanf("%lf,%lf",&x,&y)==2&&SIZE>size){
        //printf("hi");
        //printf("%lf,%lf",x,y);
        a[2*size]=x;
        a[2*size+1]=y;
        size++;
    }
    double m=0,c=0;
    int iter=10;
    double lr=0.0001;
    double* recieve=malloc(2*sizeof(double));
    while(iter--){
        recieve=gradient_descent(m,c,lr,size,a,recieve);
        m=recieve[0];
        c=recieve[1];
    }
        printf("m=%lf,c=%lf",m,c);
}