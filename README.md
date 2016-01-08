# python_review_prediction
Final classwork: Predict restaurant reviews

Following scripts are used to generated finalCode.py. This scripts contains code as well as svm model fit trained over a huge data set. Script 
when run, reads itself and creates pickled fit files. Then it load them in the model fit objects and thus we could use it to predict without
retraining our model again and again (which takes around 5-6 hours).

Scripts:  
**svm.py svm inline.py**: Creates model fits from large set of training data and dump these fits in working directory.  
**template.py**: This is only used to read and prepare final code's from.  
**codeGenerator.py**: Read all the pickle files in the current directory and generate the finalCode.py by including these fits and template.py.  
**voting.py**: Another script which employs ensemble model.
