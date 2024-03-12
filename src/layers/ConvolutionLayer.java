package layers;

import data.MatrixUtility;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer extends Layer{

    private final long SEED;
    private List<double[][]> filters;
    private int filterSize;
    private int stepSize;
    private int inLength;
    private int inRows;
    private int inCols;
    private List<double[][]> lastInput;
    private double learningRate;
    public ConvolutionLayer(int filterSize, int stepSize, int inLength, int inRows, int inCols, long SEED, int numFilters, double learningRate) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
        this.SEED = SEED;
        this.learningRate = learningRate;

        generateFilters(numFilters);
    }
    private void generateFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random r = new Random(SEED);

        for(int n=0;n<numFilters;n++){
            double[][] newFilter = new double[filterSize][filterSize];

            for(int i=0;i<filterSize;i++){
                for(int j=0;j<filterSize;j++){

                    double val = r.nextGaussian();
                    newFilter[i][j] = val;
                }
            }

            filters.add(newFilter);

        }

        this.filters = filters;
    }
    public List<double[][]> forward(List<double[][]>list){
        lastInput=list;

        List<double[][]> output = new ArrayList<>();

        for(int n=0;n<list.size();n++){
            for(double[][] filter:filters){
                output.add(convolve(list.get(n),filter,stepSize));
            }
        }

        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {

        int outRows = (input.length-filter.length)/stepSize + 1;
        int outCols = (input[0].length-filter[0].length)/stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;

        for(int i=0;i<=inRows-fRows;i+=stepSize){

            int outCol = 0;

            for(int j=0;j<=inCols-fCols;j+=stepSize){

                double sum=0;

                //APPLY FILTER AROUND THIS POSITION
                for(int x=0;x<fRows;x++){
                    for(int y=0;y<fCols;y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                        sum+=value;
                    }
                }

                output[outRow][outCol++] = sum;
            }
            outRow++;
        }

        return output;
    }
    private double[][] spaceArray(double[][] input){

        if(stepSize==1)return input;

        int outRows = (input.length-1)*stepSize +1;
        int outCols = (input[0].length)*stepSize +1;

        double[][] output = new double[outRows][outCols];

        for(int i=0;i<input.length;i++){
            for(int j=0;j<input[0].length;j++){
                output[i*stepSize][j*stepSize] = input[i][j];
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = forward(input);
        return nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input,inLength,inRows,inCols);
        return getOutput(matrixInput);
    }

    @Override
    public void backward(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();

        for(int f=0;f<filters.size();f++){
            filtersDelta.add(new double[filterSize][filterSize]);
        }

        for(int i=0;i<lastInput.size();i++){

            double[][] errorForInput = new double[inRows][inCols];

            for(int f=0;f<filters.size();f++){

                double[][] currFilter = filters.get(f);
                double[][] error = dLdO.get(i*filters.size()+f);

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(lastInput.get(i),spacedError,1);

                double[][] delta = MatrixUtility.multiply(dLdF,learningRate*-1);
                double[][] newTotalDelta = MatrixUtility.add(filtersDelta.get(f),delta);

                filtersDelta.set(f,newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = MatrixUtility.add(errorForInput,fullConvolve(currFilter,flippedError));
            }

            dLdOPreviousLayer.add(errorForInput);
        }

        for(int f=0;f<filters.size();f++){
            double[][] modified = MatrixUtility.add(filtersDelta.get(f),filters.get(f));
            filters.set(f,modified);
        }

        if(previousLayer!=null){
            previousLayer.backward(dLdOPreviousLayer);
        }

    }
    public double[][] flipArrayHorizontal(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                output[rows-i-1][j] = array[i][j];
            }
        }

        return output;
    }
    public double[][] flipArrayVertical(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                output[i][cols-j-1] = array[i][j];
            }
        }

        return output;
    }
    private double[][] fullConvolve(double[][] input, double[][] filter) {

        int outRows = (input.length+filter.length) + 1;
        int outCols = (input[0].length+filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;

        for(int i=-fRows+1;i<inRows;i++){

            int outCol = 0;

            for(int j=-fCols+1;j<inCols;j++){

                double sum=0;

                //APPLY FILTER AROUND THIS POSITION
                for(int x=0;x<fRows;x++){
                    for(int y=0;y<fCols;y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;
                        if(inputRowIndex>=0 && inputColIndex>=0 && inputRowIndex<inRows && inputColIndex<inCols){
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum+=value;
                        }
                    }
                }

                output[outRow][outCol++] = sum;
            }
            outRow++;
        }

        return output;
    }
    @Override
    public void backward(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO,inLength,inRows,inCols);
        backward(matrixInput);
    }

    @Override
    public int getOutputLength() {
        return filters.size()*inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows-filterSize)/stepSize +1;
    }

    @Override
    public int getOutputCols() {
        return (inCols-filterSize)/stepSize +1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols()*getOutputRows()*getOutputLength();
    }
}
