package my_pack;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.*;

public class TransactionsRandomForestModel {
	
	private static Classifier rf_classifier;
	private static Instances training_instances;
	
	public static void trainAndEvaluateModel(String train_data_path, String model_file_name) throws Exception {
		
		// STEP 1: Creating Training data instances.
		File inputFile = new File(train_data_path);
		CSVLoader csv_loader = new CSVLoader();   
		try {
			csv_loader.setFile(inputFile);
			training_instances = csv_loader.getDataSet();
		} catch (IOException e) {
				System.out.println("Error loading the dataset.");
				e.printStackTrace();
			}
		training_instances.setClassIndex((training_instances.numAttributes() - 1));

		// STEP 2: Constructing and building Model
		rf_classifier = new RandomForest();
		rf_classifier.buildClassifier(training_instances);

		// STEP 3: Saving the model in the saved_models directory
		weka.core.SerializationHelper.write("saved_models\\" + model_file_name, rf_classifier);
		weka.core.SerializationHelper.write("saved_models\\training_instances_class_attribute.obj", training_instances.classAttribute());
		System.out.println("Model Trained and Saved to \"saved_models\" Successfully.");

		// STEP 4: Evaluating using k-fold cross-validation
		Evaluation evaluation = new Evaluation(training_instances);
		evaluation.crossValidateModel(rf_classifier, training_instances, 10, new Random(1));

		// STEP 5: Displaying the results  
		System.out.println(rf_classifier);
		System.out.println(evaluation.toSummaryString("\nResults\n======\n", true));
		System.out.println(evaluation.toClassDetailsString());
		System.out.println(evaluation.toMatrixString());
	}
}
