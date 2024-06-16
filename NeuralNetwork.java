import java.io.*;
import java.nio.ByteBuffer;
import java.util.Random;

public class NeuralNetwork{
	private String labelFile, imagesFile;
	private double[][] labels, images;
	private int iw, ih;
	
	private double[][] weights1, weights2, bias1, bias2;
	
	private final int hiddenLayerNeurons = 30;
	private final int outputNeurons = 10;
	private final double learnRate = 0.01;
	private final int epochs = 1;

	public NeuralNetwork(String lfile, String ifile){
		this.labelFile = lfile;
		this.imagesFile = ifile;
		
		loadLabels();
		loadImages();
		
		this.weights1 = new double[hiddenLayerNeurons][this.iw*this.ih];
		this.weights2 = new double[outputNeurons][hiddenLayerNeurons];
		fill(weights1);
		fill(weights2);
		
		this.bias1 = new double[hiddenLayerNeurons][1];
		this.bias2 = new double[outputNeurons][1];
		
		// Load training
		File file = new File("training.txt");
		if (file.exists()){
			try {
				BufferedReader reader = new BufferedReader(new FileReader(file));
				weights1 = readMatrix(reader);
				weights2 = readMatrix(reader);
				bias1 = readMatrix(reader);
				bias2 = readMatrix(reader);
				reader.close();
				System.out.println("Training loaded");
				try { Thread.sleep(1000); } catch (InterruptedException ex){}
			} catch (IOException ex){
				ex.printStackTrace();
			}
		}
		
		System.out.println("Data loaded");
	}
	
	public void train(){
		for (int epoch = 0; epoch < epochs; epoch++){
			int correct = 0;
			for (int i = 0; i < this.images.length; i++){
				boolean result = trainImage(this.images[i], this.labels[i]);
				if (result){
					correct++;
				}
				
				//System.out.println("Done image "+i);
				System.out.format("%.2f %.2f\n", (double)correct/this.images.length*100, (double)i/this.images.length*100);
			}
			
			System.out.format("Accuracy: %.2f\n", (double)correct/this.images.length*100);
			try { Thread.sleep(1000); } catch (InterruptedException ex){}
		}
		
		saveTraining("training.txt");
	}
	
	public boolean trainImage(double[] image, double[] label){
		boolean result = false;
		double[][] img = makeMatrix(image);
		double[][] l = makeMatrix(label);

		// Forward propagation
		double[][] h = sum(bias1, multiply(weights1, img));
		activationFunction(h);
		double[][] o = sum(bias2, multiply(weights2, h));
		activationFunction(o);

		if (maxIndex(o) == maxIndex(l)){
			result = true;
		}
		
		double[][] deltaO = subtract(o, l);
		weights2 = sum(weights2, multiply(multiply(deltaO, transpose(h)), -learnRate));
		bias2 = sum(bias2, multiply(deltaO, -learnRate));
		
		double[][] mat1 = multiply(transpose(weights2), deltaO);
		double[][] mat2 = derivative(h);
		double[][] deltaH = new double[mat1.length][1];
		for (int j = 0; j < deltaH.length; j++){
			deltaH[j][0] = mat1[j][0]*mat2[j][0];
		}
		
		weights1 = sum(weights1, multiply(multiply(deltaH, transpose(img)), -learnRate));
		bias1 = sum(bias1, multiply(deltaH, -learnRate));
		
		return result;
	}
	
	public double[] getImage(int index){
		return this.images[index];
	}
	
	public double[][] recognize(double[] image){
		double[][] img = makeMatrix(image);

		// Forward propagation
		double[][] h = sum(bias1, multiply(weights1, img));
		activationFunction(h);
		double[][] o = sum(bias2, multiply(weights2, h));
		activationFunction(o);

		return o;
	}
	
	public void saveTraining(String fileName){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(fileName)));
			writeMatrix(writer, this.weights1);
			writeMatrix(writer, this.weights2);
			writeMatrix(writer, this.bias1);
			writeMatrix(writer, this.bias2);
			writer.close();
			System.out.println("Training saved");
		} catch (IOException ex){
			ex.printStackTrace();
		}
	}
	
	private void writeMatrix(BufferedWriter writer, double[][] mat) throws IOException{
		writer.write(String.format("%dx%d\n", mat.length, mat[0].length));
		for (int i = 0; i < mat.length; i++){
			for (int j = 0; j < mat[0].length; j++){
				writer.write(mat[i][j]+" ");
			}
			writer.newLine();
		}
	}
	
	private double[][] readMatrix(BufferedReader reader) throws IOException{
		String line = reader.readLine();
		int w = Integer.parseInt(line.split("x")[0]);
		int h = Integer.parseInt(line.split("x")[1]);
		double[][] output = new double[w][h];
		for (int i = 0; i < w; i++){
			String line2 = reader.readLine();
			for (int j = 0; j < h; j++){
				output[i][j] = Double.parseDouble(line2.split(" ")[j]);
			}
		}

		return output;
	}
	
	private double[][] makeMatrix(double[] vector){
		double[][] output = new double[vector.length][1];
		
		for (int i = 0; i < output.length; i++){
			output[i][0] = vector[i];
		}
		
		return output;
	}
	
	private double[][] derivative(double[][] h){
		double[][] output = new double[h.length][h[0].length];
		
		for (int i = 0; i < output.length; i++){
			for (int j = 0; j < output[i].length; j++){
				output[i][j] = h[i][j] * (1 - h[i][j]);
			}
		}
		
		return output;
	}
	
	public int maxIndex(double[][] matrix){
		int idx = -1;
		double max = Double.NEGATIVE_INFINITY;
		
		for (int i = 0; i < (matrix.length == 1 ? matrix[0].length : matrix.length); i++){
			double ele = matrix.length == 1 ? matrix[0][i] : matrix[i][0];
			if (ele > max){
				max = ele;
				idx = i;
			}
		}
		
		return idx;
	}
	
	private void activationFunction(double[][] matrix){
		for (int i = 0; i < matrix.length; i++){
			for (int j = 0; j < matrix[i].length; j++){
				matrix[i][j] = 1 / (1 + Math.exp(-matrix[i][j]));
			}
		}
	}
	
	private void fill(double[][] matrix){
		Random random = new Random();
		for (int i = 0; i < matrix.length; i++){
			for (int j = 0; j < matrix[i].length; j++){
				matrix[i][j] = random.nextDouble()-0.5;
			}
		}
	}
	
	private void loadLabels(){
		InputStream inputStream = null;
		try {
			File file = new File(this.labelFile);
			inputStream = new FileInputStream(file);
			byte[] buffer = new byte[(int)file.length()];
			if (inputStream.read(buffer) == -1){
				throw new IOException("Error");
			}
			
			int items = ByteBuffer.wrap(buffer, 4, 4).getInt();
			this.labels = new double[items][10];
			for (int i = 0; i < items; i++){
				int label = buffer[8+i] & 0xFF;
				this.labels[i][label] = 1.0;
			}
		} catch (IOException ex){
			ex.printStackTrace();
		} finally {
			if (inputStream != null){
				try {
					inputStream.close();
				} catch (IOException ex){
					ex.printStackTrace();
				}
			}
		}
	}
	
	private void loadImages(){
		InputStream inputStream = null;
		try {
			File file = new File(this.imagesFile);
			inputStream = new FileInputStream(file);
			byte[] buffer = new byte[(int)file.length()];
			if (inputStream.read(buffer) == -1){
				throw new IOException("Error");
			}
			
			int items = ByteBuffer.wrap(buffer, 4, 4).getInt();
			int rows = ByteBuffer.wrap(buffer, 8, 4).getInt();
			int columns = ByteBuffer.wrap(buffer, 12, 4).getInt();
			this.iw = columns;
			this.ih = rows;
			this.images = new double[items][rows*columns];
			for (int i = 0; i < items; i++){
				for (int j = 0; j < rows*columns; j++){
					int pixel = buffer[16+i*rows*columns+j] & 0xFF;
					this.images[i][j] = pixel != 0 ? 1 : 0; ///255.0;
				}
			}
		} catch (IOException ex){
			ex.printStackTrace();
		} finally {
			if (inputStream != null){
				try {
					inputStream.close();
				} catch (IOException ex){
					ex.printStackTrace();
				}
			}
		}
	}
	
	private void printImage(int n){
		double[] image = this.images[n];
		for (int i = 0; i < this.ih; i++){
			for (int j = 0; j < this.iw; j++){
				double p = image[i*this.iw+j];
				System.out.print(p > 0 ? "#" : ".");
			}
			System.out.println();
		}
	}
	
	private static double[][] multiply(double[][] mat1, double[][] mat2){
		if (mat1[0].length != mat2.length){
			throw new ArithmeticException(String.format("Incompatible size: %dx%d and %dx%d", mat1.length, mat1[0].length, mat2.length, mat2[0].length));
		}
		double[][] output = new double[mat1.length][mat2[0].length];
		
		for (int i = 0; i < output.length; i++){
			for (int j = 0; j < output[0].length; j++){
				double sum = 0;
				for (int k = 0; k < mat1[0].length; k++){
					sum += mat1[i][k]*mat2[k][j];
				}
				output[i][j] = sum;
			}
		}
		
		return output;
	}
	
	private static double[][] multiply(double[][] matrix, double k){
		double[][] output = new double[matrix.length][matrix[0].length];
		
		for (int i = 0; i < output.length; i++){
			for (int j = 0; j < output[i].length; j++){
				output[i][j] = matrix[i][j]*k;
			}
		}
		
		return output;
	}
	
	private static double[][] sum(double[][] a, double[][] b){
		if (a.length != b.length || a[0].length != b[0].length){
			throw new ArithmeticException(String.format("Incompatible size: %dx%d and %dx%d", a.length, a[0].length, b.length, b[0].length));
		}
		double[][] output = new double[a.length][a[0].length];
		
		for (int i = 0; i < output.length; i++){
			for (int j = 0; j < output[i].length; j++){
				output[i][j] = a[i][j]+b[i][j];
			}
		}
		
		return output;
	}
	
	private static double[][] subtract(double[][] a, double[][] b){
		return sum(a, multiply(b, -1));
	}
	
	private static double[][] transpose(double[][] matrix){
		double[][] output = new double[matrix[0].length][matrix.length];
		
		for (int i = 0; i < output.length; i++){
			for (int j = 0; j < output[0].length; j++){
				output[i][j] = matrix[j][i];
			}
		}
		
		return output;
	}
	
	private static void printMatrix(String name, double[][] matrix){
		System.out.println("----"+name);
		for (int i = 0; i < matrix.length; i++){
			for (int j = 0; j < matrix[0].length; j++){
				System.out.format("%.5f ", matrix[i][j]);
			}
			System.out.println();
		}
		System.out.println("-------");
	}

	public static void main(String[] args){
		NeuralNetwork nn = new NeuralNetwork("train-labels-idx1-ubyte", "train-images-idx3-ubyte");
		nn.train();
		//nn.printImage(0);
	}
}