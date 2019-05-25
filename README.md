# Disaster_Response_Pipelines
The Repository contains all files to run the Disaster Response classification as a web app.


Execution
---------
- 1. Data/process_data.py
	prepares the necessary data
	Input: 2 csv-files (see example execution below)
	Output: sql-lite database (see example execution below) containing table 'InsertTableName'

	terminal example execution:
	"python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db"

- 2. models/train_classifier.py
	trains the model
	Input: sql-lite database from step 1 (see example execution below)
	       path...str variable which has to be assigned in the first line of the code; contains the the name of the project 	       		folder, e.g. "/py1150/Disaster_Response_Pipelines/"
			the default path currently stored is '/home/bernd/Documents/Python/Disaster_Response_Pipelines/' --> 				substitute with individual local path
	Output: pkl file containing model (see example execution below)

	terminal example execution: 
	"python3 train_classifier.py /Data/DisasterResponse.db classifier.pkl"
	
- 3. app/run.py 
	executes web app
	Input: sql-lite database from step 1 (see example execution below)
	       pkl file containing model (see example execution below)

	terminal example execution:
	"python3 run.py /Data/DisasterResponse.db /models classifier.pkl"

