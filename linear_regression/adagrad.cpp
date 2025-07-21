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
    double inter;
    double sl;
    double learn;
    int epochs;
    double loss_function;
    double rms_grad;

    vector<double>ind_var;
    vector<double>dep_var;
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
                rms_grad+=s_grad*s_grad+i_grad*i_grad;
                sl-=learn*s_grad/sqrt(rms_grad);
                inter-=learn*i_grad/sqrt(rms_grad);
            }
        }
    }

    public:
    LinearRegression(){
        this->learn=0.001;
        this->epochs=100;
        this->batch_size=256;
    }
    LinearRegression(double lr,int iter,int batch){
        this->learn=lr;
        this->epochs=iter;
        this->batch_size=batch;
    }
    void fit(vector<double>&x,vector<double>&y){
        ind_var.clear();
        dep_var.clear();
        this->inter=0;
        this->sl=0;
        this->size=x.size();
        this->batch_size=min(this->batch_size,this->size);
        this->rms_grad=0;
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
    LinearRegression lr(0.1,10,10);
    lr.fit(x,y);
    cout<<lr.loss()<<endl;
    cout<<lr.slope()<<endl;
    cout<<lr.intercept()<<endl;
}
