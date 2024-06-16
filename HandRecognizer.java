import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.canvas.*;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.animation.AnimationTimer;

import java.io.*;
import java.nio.ByteBuffer;

public class HandRecognizer extends Application{
	private static final int WIDTH = 28*20+400, HEIGHT = 28*20;
	private static final Font FONT = new Font("sans-serif", 25);
	private static final int TRAINING_IMAGES = 20;

	private boolean[][] map = new boolean[28][28];
	private double[][] currentEvaluation;
	private NeuralNetwork nn;

	@Override
	public void start(Stage stage){
		StackPane pane = new StackPane();


		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		GraphicsContext gc = canvas.getGraphicsContext2D();
		pane.getChildren().add(canvas);
		
		this.nn = new NeuralNetwork("train-labels-idx1-ubyte", "train-images-idx3-ubyte");
		
		canvas.setFocusTraversable(true);
		canvas.setOnKeyPressed(e -> {
			if (e.getCode() == KeyCode.DELETE){
				this.map = new boolean[28][28];
				this.currentEvaluation = null;
			} else if (e.getCode() == KeyCode.L){
				double[] img = this.nn.getImage((int)(Math.random()*59999));
				for (int i = 0; i < 28; i++){
					for (int j = 0; j < 28; j++){
						map[i][j] = img[j*28+i] > 0;
					}
				}
				evaluate();
			}
			
			if (this.currentEvaluation != null){
				try {
					int digit = Integer.parseInt(e.getText());
					double[] img = generateCurrentImage();
					double[] label = new double[10];
					label[digit] = 1.0;
					
					boolean result = this.nn.trainImage(img, label);
					System.out.println("Trained network for "+digit+" with result: "+result);
					this.nn.saveTraining("training.txt");
					this.map = new boolean[28][28];
					this.currentEvaluation = null;
				} catch (NumberFormatException ex){
				}
			}
		});
		
		canvas.setOnMouseDragged(e -> {
			int x = (int)(e.getX()/20.0);
			int y = (int)(e.getY()/20.0);
			if (e.getButton() == MouseButton.PRIMARY){
				for (int i = x-1; i <= x+1; i++){
					for (int j = y-1; j <= y+1; j++){
						if (i >= 0 && j >= 0 && i < 28 && j < 28){
							this.map[i][j] = true;
						}
					}
				}
				evaluate();
			} else if (e.getButton() == MouseButton.SECONDARY){
				this.map[x][y] = false;
			}
		});
		
		AnimationTimer loop = new AnimationTimer(){
			@Override
			public void handle(long time){
				update(gc);
			}
		};
		loop.start();
		
		stage.setTitle("Handwritten digits recognizer");
		stage.setScene(new Scene(pane, WIDTH, HEIGHT));
		stage.show();
	}

	private double[] generateCurrentImage(){
		double[] img = new double[28*28];
		for (int i = 0; i < 28; i++){
			for (int j = 0; j < 28; j++){
				img[28*j+i] = this.map[i][j] ? 1.0 : 0.0;
			}
		}

		return img;
	}
	
	private void evaluate(){
		double[] img = generateCurrentImage();
		this.currentEvaluation = this.nn.recognize(img);
	}
	
	private void update(GraphicsContext gc){
		gc.clearRect(0, 0, WIDTH, HEIGHT);
		
		for (int i = 0; i < 28; i++){
			for (int j = 0; j < 28; j++){
				gc.setFill(this.map[i][j] ? Color.BLACK : Color.WHITE);
				gc.fillRect(i*20, j*20, 20, 20);
			}
		}
		
		gc.setFont(FONT);
		int idx = this.currentEvaluation == null ? -1 : this.nn.maxIndex(this.currentEvaluation);
		for (int i = 0; i < 10; i++){
			gc.setFill(i == idx ? Color.BLUE : Color.RED);
			gc.fillText(Integer.toString(i), 580, 70+i*50);
			gc.setFill(Color.LIME);
			gc.fillRect(610, 50+i*50, 200*(this.currentEvaluation == null ? 0 : this.currentEvaluation[i][0]), 25);
			gc.strokeRect(610, 50+i*50, 200, 25);
			gc.setFill(Color.BLACK);
			gc.fillText(String.format("%.2f%%", this.currentEvaluation == null ? 0 : this.currentEvaluation[i][0]*100), 820, 70+i*50);
		}
	}

	public static void main(String[] args){
		launch(args);
	}
}