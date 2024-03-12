package network;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    private NeuralNetwork network;
    private int inputRows;
    private int inputCols;
    private final double scaleFactor;
    List<Layer> layers;

    public NetworkBuilder(int inputRows, int inputCols,double scaleFactor) {
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.scaleFactor = scaleFactor;
        layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters,int filterSize,int stepSize,double learningRate, long SEED){
        if(layers.isEmpty()){
            layers.add(new ConvolutionLayer(filterSize,stepSize,1,inputRows,inputCols,SEED,numFilters,learningRate));
        }else{
            Layer layerPrev = layers.get(layers.size()-1);
            layers.add(new ConvolutionLayer(filterSize,stepSize,layerPrev.getOutputLength(),layerPrev.getOutputRows(),layerPrev.getOutputCols(),SEED,numFilters,learningRate));
        }
    }

    public void addMaxPoolLayer(int windowSize,int stepSize){
        if(layers.isEmpty()){
            layers.add(new MaxPoolLayer(stepSize,windowSize,1,inputRows,inputCols));
        }else{
            Layer layerPrev = layers.get(layers.size()-1);
            layers.add(new MaxPoolLayer(stepSize,windowSize,layerPrev.getOutputLength(),layerPrev.getOutputRows(),layerPrev.getOutputCols()));
        }
    }

    public void addFullConnectLayer(int outLength,double learningRate, long SEED){
        if(layers.isEmpty()){
            layers.add(new FullyConnectedLayer(inputRows*inputCols,outLength,SEED,learningRate));
        }else{
            Layer layerPrev = layers.get(layers.size()-1);
            layers.add(new FullyConnectedLayer(layerPrev.getOutputElements(),outLength,SEED,learningRate));
        }
    }

    public NeuralNetwork build(){
        network = new NeuralNetwork(layers,scaleFactor);
        return network;
    }
}
