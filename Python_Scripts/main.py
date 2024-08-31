import os
from flask import Flask, request, Response
from flask_cors import CORS

# Import your custom modules
from trainingModel import Model_training
from predictFromModel import prediction

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predictRouteClient():
    try:
        print("Received request at /predict")
        folder_path = request.json.get('folderPath')
        
        if folder_path:
            # Create an instance of the prediction class
            pred = prediction()
            # Predicting for dataset present in the folder
            print(f"Predicting for folder: {folder_path}")
            path = pred.predictionFromModel(folder_path)
            
            return Response(f"Prediction File created at {path}!!!", status=200)
        
        else:
            return Response("folderPath not provided in the request", status=400)
    
    except ValueError as e:
        return Response(f"Error Occurred! ValueError: {str(e)}", status=400)
    
    except KeyError as e:
        return Response(f"Error Occurred! KeyError: {str(e)}", status=400)
    
    except Exception as e:
        return Response(f"Error Occurred! {str(e)}", status=500)

@app.route("/train", methods=['POST'])
def trainRouteClient():
    try:
        print("Received request at /train")
        folder_path = request.json.get('folderPath')
        
        if folder_path:
            # Create an instance of the Model_training class
            trainer = Model_training()
            # Training the model for the files in the folder
            print(f"Training for folder: {folder_path}")
            trainer.trainingModel(folder_path)
            
            return Response("Training successful!!", status=200)
        
        else:
            return Response("folderPath not provided in the request", status=400)

    except ValueError as e:
        return Response(f"Error Occurred! ValueError: {str(e)}", status=400)

    except KeyError as e:
        return Response(f"Error Occurred! KeyError: {str(e)}", status=400)

    except Exception as e:
        return Response(f"Error Occurred! {str(e)}", status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)