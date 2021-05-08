package my_pack;

import java.io.*;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class FraudPredictionMain {

	public static void main(String[] args) throws Exception {
		
		
		/* A. Training, Evaluating and Saving the model. */ 
		/*
		String training_dataset = 
			"C:\\Users\\nag\\Desktop\\PROJECTS\\projects\\cc_fraud_prediction\\dataset\\creditcard_nominal_reduced.csv";
			
		TransactionsRandomForestModel.trainModel(training_dataset, "creditcard_fraud.model");
		System.out.println("Training Complete");
		TransactionsRandomForestModel.printStats(); 
		System.out.println("Evaluation Complete");
		*/
		
		/*----------------------------------------------------------------------------------------------------------------------*/
		
		/* B. Loading the saved Model Object for making the predictions. */
		Classifier rf_classifier = null;
		try {
			rf_classifier = (Classifier) weka.core.SerializationHelper.read("saved_models\\creditcard_fraud.model");
		}catch (Exception e) {
			System.out.println("Unable to load the classifier.");
			e.printStackTrace();
		}
		
		/*----------------------------------------------------------------------------------------------------------------------*/
		
		/* C. Creating Unlabeled Data instances */
		String unlabeled_data = "C:\\Users\\nag\\Desktop\\PROJECTS\\projects\\cc_fraud_prediction\\dataset\\unlabeled_data2.arff";
        File inputFile = new File(unlabeled_data);
        ArffLoader atf = new ArffLoader();   
        atf.setFile(inputFile);
        Instances unlabeled_id_instances = atf.getDataSet();
        unlabeled_id_instances.setClassIndex((unlabeled_id_instances.numAttributes() - 1)); 
        
        // Removing Transaction ID column from the Unlabeled data
        String[] opts = new String[]{ "-R", "1"};
        Remove remove = new Remove();  
        remove.setOptions(opts);
        remove.setInputFormat(unlabeled_id_instances); 
        Instances unlabeled_instances = Filter.useFilter(unlabeled_id_instances, remove);
        
        /*----------------------------------------------------------------------------------------------------------------------*/
        
        /* D. Predicting the Fraud of the Unlabeled Data */
        for(int i=0; i<unlabeled_instances.numInstances(); i++) {
        	double s = rf_classifier.classifyInstance(unlabeled_instances.instance(i));
        	Attribute training_class_attr = (Attribute) weka.core.SerializationHelper
        															.read("saved_models\\training_instances_class_attribute.obj");
        	
        	System.out.printf("Transaction-ID: %-3d",(int)unlabeled_id_instances.get(i).value(0));
        	System.out.printf(" Transaction-Time: %-5d",(int)unlabeled_instances.get(i).value(0));
        	System.out.printf(" Amount: €%-5d Prediction: ", (int)unlabeled_instances.get(i)
        																		.value(unlabeled_instances.numAttributes()-2));
        	if(training_class_attr.value((int) s).equals("fraud-positive"))
        		System.err.println("Fraud Detected");
        	else
        		System.out.println("Normal");
        }
        
        /*----------------------------------------------------------------------------------------------------------------------*/
	}
}
