#include<iostream>
#include<fstream>
#include<string>
#include<vector>
using namespace std;

class LinearRegression{
    private:
    int size;
    double inter;
    double sl;
    double learn;
    int epochs;
    double loss_function;
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
    double slope_gradient(){
        double a=0;
        for(int i=0;i<this->size;i++){
            a+=2*(sl*ind_var[i]+inter-dep_var[i])*ind_var[i]/size;
        }
        return a;
    }
    double inter_gradient(){
        double a=0;
        for(int i=0;i<size;i++){
            a+=2*(sl*ind_var[i]+inter-dep_var[i])/size;
        }
        return a;
    }
    void gradient_descent(){
        for(int i=0;i<this->epochs;i++){
            double s_grad=slope_gradient();
            double i_grad=inter_gradient();
            sl-=learn*s_grad;
            inter-=learn*i_grad;
        }
    }
    public:
    LinearRegression(){
        this->learn=0.001;
        this->epochs=1000;
    }
    LinearRegression(double lr,int iter){
        this->learn=lr;
        this->epochs=iter;
    }
    void fit(vector<double>&x,vector<double>&y){
        ind_var.clear();
        dep_var.clear();
        this->inter=0;
        this->sl=0;
        this->size=x.size();
        for(auto i:x){
            ind_var.push_back(i);
        }
        for(auto i:y){
            dep_var.push_back(i);
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
    LinearRegression lr(0.001,1000);
    lr.fit(x,y);
    cout<<lr.loss()<<endl;
    cout<<lr.slope()<<endl;
    cout<<lr.intercept()<<endl;
}
