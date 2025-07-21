#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<algorithm>
#include<ctime>
#include<random>
using namespace std;

class LinearRegression{

    private:

    int size;
    int batch_size;
    int epochs;
    double inter;
    double sl;
    double learn;
    double loss_function;
    double beta1,beta2;
    double sl_momentum,inter_momentum,sl_rms,inter_rms;
    double beta1_factor,beta2_factor;

    vector<double>ind_var,dep_var;
    void _loss(){
        loss_function=0;
        for(int i=0;i<this->size;i++){
            double l=sl*ind_var[i]+inter-dep_var[i];
            l*=l;
            loss_function+=l/this->size;
        }
    }
    double slope_gradient(int l,int r){
        double a=0;
        for(int i=l;i<r;i++){
            a+=2*(sl*ind_var[i]+inter-dep_var[i])*ind_var[i]/(r-l);
        }
        return a;
    }
    double inter_gradient(int l,int r){
        double a=0;
        for(int i=l;i<r;i++){
            a+=2*(sl*ind_var[i]+inter-dep_var[i])/(r-l);
        }
        return a;
    }
    void gradient_descent(){
        for(int j=0;j<this->epochs;j++){
            for(int i=0;i<this->size;i+=this->batch_size){
                double s_grad=slope_gradient(i,min(this->size,this->batch_size+i));
                double i_grad=inter_gradient(i,min(this->size,this->batch_size+i));
                sl_momentum=beta1*sl_momentum+(1-beta1)*s_grad;
                inter_momentum=beta1*inter_momentum+(1-beta1)*i_grad;
                beta1_factor*=beta1;
                sl_rms=beta2*sl_rms+(1-beta2)*(s_grad*s_grad);
                inter_rms=beta2*inter_rms+(1-beta2)*(i_grad*i_grad);
                beta2_factor*=beta2;
                double s_momentum=sl_momentum/(1-beta1_factor);
                double i_momentum=inter_momentum/(1-beta1_factor);
                double s_rms=sl_rms/(1-beta2_factor);
                double i_rms=inter_rms/(1-beta2_factor);
                sl-=learn*s_momentum/(sqrt(s_rms)+1e-5);
                inter-=learn*i_momentum/(sqrt(i_rms)+1e-5);
            }
        }
    }

    public:
    LinearRegression(){
        this->learn=0.001;
        this->epochs=100;
        this->batch_size=256;
        this->beta1=0.9;
        this->beta2=0.9;
    }
    LinearRegression(double lr,int iter,int batch,double beta1,double beta2){
        this->learn=lr;
        this->epochs=iter;
        this->batch_size=batch;
        this->beta1=beta1;
        this->beta2=beta2;
    }
    void fit(vector<double>&x,vector<double>&y){
        ind_var.clear();
        dep_var.clear();
        this->inter=0;
        this->sl=0;
        this->size=x.size();
        this->batch_size=min(this->batch_size,this->size);
        this->sl_momentum=0;
        this->inter_momentum=0;
        this->sl_rms=0;
        this->inter_rms=0;
        this->beta1_factor=1;
        this->beta2_factor=1;
 
        vector<int>perm(this->size);
        for(int i=0;i<this->size;i++){
            perm[i]=i;
        }
        unsigned seed = static_cast<unsigned>(std::time(nullptr)); // or any fixed number for reproducibility
        std::default_random_engine rng(seed);
        shuffle(perm.begin(),perm.end(),rng);
        for(int i=0;i<this->size;i++){
            ind_var.push_back(x[perm[i]]);
            dep_var.push_back(y[perm[i]]);
        }
        gradient_descent();
        _loss();
    }
    double slope(){
        return this->sl;
    }
    double intercept(){
        return this->inter;
    }
    double loss(){
        return loss_function;
    }
    double predict(double x){
        return sl*x+inter;
    }
};





int main(){
    int n;
    cin>>n;
    vector<double>x(n),y(n);
    char s;
    for(int i=0;i<n;i++){
        cin>>x[i]>>s>>y[i];
    }
    LinearRegression lr(0.1,100,10,0.9,0.9);
    lr.fit(x,y);
    cout<<lr.loss()<<endl;
    cout<<lr.slope()<<endl;
    cout<<lr.intercept()<<endl;
}
