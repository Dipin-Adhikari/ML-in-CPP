#include<iostream>
#include<vector>
#include<cmath>

using namespace std;

class NeuralNetwork{
    private:
        double learningRate;
        double bias;
        vector<double> weights;
    public:
        NeuralNetwork(int noOfNeuron, double learningRate=1){
            weights.resize(noOfNeuron);
            this->learningRate = learningRate;
            for (int i=0; i<weights.size(); ++i){
                weights[i] = ((double)rand()/(RAND_MAX));
            }
            bias = ((double)rand()/(RAND_MAX)); 
        }

        vector<double> sigmoid(const vector<double> z){
            vector<double> sig;
            for (int i=0; i<z.size(); ++i){
                sig.push_back(1/(1+exp(-z[i])));
            }
            return (sig);
        }

        vector<double> sigmoidDerivative(const vector<double> z){
            vector<double> sigD;
            for(int i=0; i<z.size(); ++i){
                sigD.push_back(z[i]*(1-z[i]));
            }
            return sigD;
        }

        vector<double> forwardPass(const vector<vector<int>> &inputs){
            vector<double> sumArray;
            for (int i=0; i<inputs.size(); ++i){
                double sum = bias;
                for (int j=0; j<inputs[0].size(); ++j){
                    sum += weights[j] * inputs[i][j];
                }
                sumArray.push_back(sum);
            }
            
            return sigmoid(sumArray);
        }


        vector<double> lossFunction(const vector<double> &predictedOutput, const vector<int> &outputs){
            vector<double> error;
            for (int i=0; i<predictedOutput.size(); ++i){
                error.push_back(outputs[i] - predictedOutput[i]);
            }
            return error;
        }


        vector<vector<int>> transpose(const vector<vector<int>> &inputs){
            if (inputs.empty()){return{};}

            vector<vector<int>> transposedInput(inputs[0].size(), vector<int>(inputs.size()));
            for (int i=0; i<inputs.size(); ++i){
                for (int j=0; j<inputs[0].size(); ++j){
                    transposedInput[i][j] = inputs[j][i];
                }
            }
            return (transposedInput);
        }


        vector<double> weightUpdate(const vector<vector<int>> &inputs, const vector<double> &error, const vector<double> &predictedOutput){
            vector<double> temp;
            vector<double> weightUpdateArray(weights.size());
            vector<double> sigmoidDeri = sigmoidDerivative(predictedOutput);
            for (int i=0; i<inputs.size(); ++i){
                temp.push_back(error[i]  * sigmoidDeri[i]);             
            }

            vector<vector<int>> transposedInputs = transpose(inputs);
            for (int i = 0; i < transposedInputs.size(); ++i) {
                weightUpdateArray[i] = 0;
                for (int j = 0; j < transposedInputs[0].size(); ++j) {
                    weightUpdateArray[i] += transposedInputs[i][j] * temp[j];
                }
            }
            return weightUpdateArray;
        }

        double biasUpdate(const vector<double> &error, const vector<double> &predictedOutput){
            double biasUpd=0;
            vector<double> sigmoidDeri = sigmoidDerivative(predictedOutput);
            for (int i=0; i<error.size(); ++i){
                biasUpd += error[i] * sigmoidDeri[i];
            }
            return biasUpd;
        }




        void train(const vector<vector<int>> &inputs, const vector<int> &outputs, int trainingIterations){
            for (int iteration=0; iteration<trainingIterations; ++iteration){
                vector<double> predictedOutput = forwardPass(inputs);
                vector<double> error = lossFunction(predictedOutput, outputs);
        
                vector<double> w = weightUpdate(inputs, error, predictedOutput);
                for (int i=0; i<weights.size(); ++i){
                    weights[i] += w[i];
                }
                bias += biasUpdate(error, predictedOutput);
                
                double meanLoss = 0.0;
                
                for (int i=0; i<error.size(); ++i){
                    meanLoss += pow(error[i], 2);
                }
                meanLoss /= error.size();
                if (iteration % 10==0){
                    cout<<"Loss = "<<meanLoss<<endl;
                }

            }
        }


        vector<double> __getWeights(){
            return weights;
        }

        double __getBiases(){
            return bias;
        }
};

int main(){
    NeuralNetwork nn(3);
    vector<vector<int>> inputs{
        {1, 0, 1},
        {1, 0, 0},
        {0, 1, 1},
    };
    vector<int> outputs{1, 0, 1};
    nn.train(inputs, outputs, 100000);
    vector<vector<int>> testInput{
        {1, 1, 0},
        {1, 1, 1},
        {0, 0, 0}};
    cout<<endl<<endl;
    for(int i=0; i<testInput.size(); ++i){
        cout<<nn.forwardPass(testInput)[i]<<endl;
    }

    return 0;
}
