package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    private int stepSize;
    private int windowSize;
    private int inLength;
    private int inRows;
    private int inCols;
    private List<int[][]> maxRowLast;
    private List<int[][]> maxColLast;
    public MaxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inCols) {
        this.stepSize = stepSize;
        this.windowSize = windowSize;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
    }

    public List<double[][]> forward(List<double[][]> input){
        List<double[][]> output = new ArrayList<>();
        maxRowLast = new ArrayList<>();
        maxColLast = new ArrayList<>();

        for(int l=0;l<input.size();l++){
            output.add(pool(input.get(l)));
        }

        return output;
    }

    public double[][] pool(double[][] input){

        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for(int r=0;r<getOutputRows();r+=stepSize){
            for(int c=0;c<getOutputCols();c+=stepSize){

                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                double max = 0;
                //SLIDING WINDOW(kernel)
                for(int x=0;x<windowSize;x++){
                    for(int y=0;y<windowSize;y++){
                        if(max < input[r+x][c+y]){
                            max = input[r+x][c+y];
                            //WHERE THE MAX VALUES CAME FROM
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }
                    }
                }

                output[r][c] = max;
            }
        }

        maxRowLast.add(maxRows);
        maxColLast.add(maxCols);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = forward(input);
        return nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixlist = vectorToMatrix(input,inLength,inRows,inCols);
        return nextLayer.getOutput(matrixlist);
    }

    @Override
    public void backward(List<double[][]> dLdO) {

        List<double[][]> dXdL = new ArrayList<>();

        int l=0;
        for(double[][] array:dLdO){

            double[][] error = new double[inRows][inCols];

            for(int r=0;r<getOutputRows();r++){
                for(int c=0;c<getOutputCols();c++){

                    int maxI = maxRowLast.get(l)[r][c];
                    int maxJ = maxColLast.get(l)[r][c];

                    if(maxI != -1){
                        error[maxI][maxJ] += array[r][c];
                    }
                }
            }

            dXdL.add(error);
            l++;

        }

        if(previousLayer!=null){
            previousLayer.backward(dXdL);
        }
    }

    @Override
    public void backward(double[] dLdO) {
        List<double[][]> matrixlist = vectorToMatrix(dLdO,getOutputLength(),getOutputRows(),getOutputCols());
        backward(matrixlist);
    }

    @Override
    public int getOutputLength() {
        return inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows-windowSize)/stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols-windowSize)/stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return inLength*getOutputRows()*getOutputCols();
    }
}
