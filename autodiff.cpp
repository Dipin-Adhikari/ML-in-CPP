#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <cmath>


using namespace std;

class VariableImpl {
public:
    double value;
    double grad;
    function<void()> _backward;
    vector<shared_ptr<VariableImpl>> _parents;
    bool visited;

    VariableImpl(double value){
        this->value = value;
        grad = 0.0;
        _backward = [](){};
        visited = false;
    }
};

class Variable {
private:
    shared_ptr<VariableImpl> impl;
    explicit Variable(shared_ptr<VariableImpl> impl){
        this->impl = move(impl);
    }

public:
    Variable(double value){
        impl = make_shared<VariableImpl>(value);
    }

    double getValue(){ 
        return impl->value; 
    }

    double getGrad(){
        return impl->grad;
    }

    Variable operator+(const Variable& other){
        auto out = make_shared<VariableImpl>(impl->value + other.impl->value);

        out->_parents.push_back(impl);
        out->_parents.push_back(other.impl);
        out->_backward = [out, this_impl=impl, other_impl=other.impl]() {
            this_impl->grad += out->grad;
            other_impl->grad += out->grad;
        };

        return Variable(out);
    }

    friend Variable operator +(const double n, const Variable& other){
        return Variable(n) + other;
    }


    Variable operator-(const Variable& other){
        auto out = make_shared<VariableImpl>(impl->value - other.impl->value);

        out->_parents.push_back(impl);
        out->_parents.push_back(other.impl);
        out->_backward = [out, this_impl=impl, other_impl=other.impl]() {
            this_impl->grad += out->grad;
            other_impl->grad -= out->grad;
        };

        return Variable(out);
    }

    friend Variable operator -(const double n, const Variable& other){
        return Variable(n) - other;
    }

    Variable operator-(){ 
        auto out = make_shared<VariableImpl>(-impl->value);
        out->_parents = {impl};
        out->_backward = [out, this_impl=impl]() {
            this_impl->grad -= out->grad;
        };
        return Variable(out);
    }

    Variable operator*(const Variable& other) {
        auto out = make_shared<VariableImpl>(impl->value * other.impl->value);

        out->_parents.push_back(impl);
        out->_parents.push_back(other.impl);
        out->_backward = [out, this_impl=impl, other_impl=other.impl]() {
            this_impl->grad += other_impl->value * out->grad;
            other_impl->grad += this_impl->value * out->grad;
        };

        return Variable(out);
    }

    friend Variable operator *(const double n, const Variable& other){
        return Variable(n) * other;
    }

    Variable operator/(const Variable& other) {
        auto out = make_shared<VariableImpl>(impl->value / other.impl->value);

        out->_parents.push_back(impl);
        out->_parents.push_back(other.impl);
        out->_backward = [out, this_impl=impl, other_impl=other.impl]() {
            this_impl->grad += (1 / other_impl->value) * out->grad;
            other_impl->grad -= (this_impl->value / (other_impl->value * other_impl->value)) * out->grad;
        };

        return Variable(out);
    }

    friend Variable operator /(const double n, const Variable& other){
        return Variable(n) / other;
    }



    Variable power(const Variable& other){
        auto out = make_shared<VariableImpl>(pow(impl->value, other.impl->value));
        out->_parents.push_back(impl);
        out->_parents.push_back(other.impl);
        out->_backward = [out, this_impl=impl, other_impl=other.impl](){
            this_impl->grad += other_impl->value * pow(this_impl->value, (other_impl->value-1)) * out->grad;
            other_impl->grad += pow(this_impl->value, other_impl->value) * log(this_impl->value) * out->grad;
        };
        return Variable(out);
    }

    Variable sine(){
        auto out = make_shared<VariableImpl>(sin(impl->value));

        out->_parents.push_back(impl);
        out->_backward = [out, this_impl=impl]() {
            this_impl->grad += cos(this_impl->value) * out->grad;
        };

        return Variable(out);
    }


    Variable cosine(){
        auto out = make_shared<VariableImpl>(cos(impl->value));

        out->_parents.push_back(impl);
        out->_backward = [out, this_impl=impl]() {
            this_impl->grad -= sin(this_impl->value) * out->grad;
        };

        return Variable(out);
    }

    Variable tangent(){
        auto out = make_shared<VariableImpl>(tan(impl->value));

        out->_parents.push_back(impl);
        out->_backward = [out, this_impl=impl]() {
            this_impl->grad += (1 / pow(cos(this_impl->value), 2)) * out->grad;
        };

        return Variable(out);
    }

    Variable exponential(){
        auto out = make_shared<VariableImpl>(exp(impl->value));

        out->_parents.push_back(impl);
        out->_backward = [out, this_impl=impl, out_impl=out]() {
            this_impl->grad += out_impl->value * out->grad;
        };

        return Variable(out);
    }

    Variable logarithm(){
        auto out = make_shared<VariableImpl>(log(impl->value));

        out->_parents.push_back(impl);
        out->_backward = [out, this_impl=impl, out_impl=out](){
            this_impl->grad += (1 / this_impl->value) * out->grad;
        };
        return Variable(out);
    }


    void backward() {
        impl->grad = 1.0;
        vector<shared_ptr<VariableImpl>> funcs = { impl };

        while (!funcs.empty()) {
            auto f = funcs.back();
            funcs.pop_back();
            if (f->visited){
                continue;
            }
            f->visited = true;

            if (f->_backward) f->_backward();

            for (const auto& parent : f->_parents) {
                funcs.push_back(parent);
            }
        }
    }
};

int main() {
    Variable x(2); 
    Variable y(3);
    Variable z(5);
    Variable f = x/y - x.tangent() * y.cosine() + x.exponential() + z.sine() * x.sine() + z.exponential()/(x.sine() + x*y*z);
    f.backward();

    cout<<"The value of f(x, y, z): "<<f.getValue()<<endl;
    cout<<"The value of gradient with respect to x: "<<x.getGrad()<<endl;
    cout<<"The value of gradient with respect to y: "<<y.getGrad()<<endl;
    cout<<"The value of gradient with respect to z: "<<z.getGrad()<<endl;

    return 0;
}
