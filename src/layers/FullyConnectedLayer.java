package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{
    private final long SEED;
    private final double leak = 0.01;
    private double learningRate;
    private double[][] weights;
    private int inLength;
    private int outLength;
    private double[] zLast;
    private double[] xLast;
    public FullyConnectedLayer(int inLength, int outLength, long SEED, double learningRate) {
        this.inLength = inLength;
        this.outLength = outLength;
        this.SEED = SEED;
        this.learningRate = learningRate;

        weights = new double[inLength][outLength];
        setRandomWeights();
    }
    public double[] forward(double[] input){
        xLast = input;
        double[] z = new double[outLength];
        double[] out = new double[outLength];

        for(int i=0;i<inLength;i++){
            for(int j=0;j<outLength;j++){
                z[j] += input[i]*weights[i][j];
            }
        }

        zLast = z;

        for(int i=0;i<inLength;i++){
            for(int j=0;j<outLength;j++){
                out[j] = reLu(z[j]);
            }
        }

        return out;
    }
    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = forward(input);

        if(nextLayer!=null){
            return nextLayer.getOutput(forwardPass);
        }
        return forwardPass;
    }

    @Override
    public void backward(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backward(vector);
    }

    @Override
    public void backward(double[] dLdO) {

        double[] dLdX = new double[inLength];

        double dOdZ;
        double dZdW;
        double dLdW;
        double dZdX;

        for(int i=0;i<inLength;i++){

            double dLdXSum = 0;

            for(int j=0;j<outLength;j++){

                dOdZ = derivativeReLu(zLast[j]);
                dZdW = xLast[i];
                dLdW = dLdO[j]*dOdZ*dZdW;
                dZdX = weights[i][j];

                weights[i][j] -= dLdW*learningRate;

                dLdXSum += dLdO[j]*dOdZ*dZdX;
            }

            dLdX[i] = dLdXSum;
        }

        if(previousLayer!=null) {
            previousLayer.backward(dLdX);
        }
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return outLength;
    }
    public void setRandomWeights(){
        Random r = new Random(SEED);

        for(int i=0;i<inLength;i++){
            for(int j=0;j<outLength;j++){
                weights[i][j] = r.nextGaussian();
            }
        }
    }
    public double reLu(double input){
        if(input <= 0){
            return 0;
        }
        return input;
    }
    public double derivativeReLu(double input){
        if(input <= 0){
            return leak;//TO PREVENT DEAD ZONES CAUSED BY HEAVISIDE FUNCTION
        }
        return 1;
    }
}
