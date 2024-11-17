# importing libraires
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
 
class Input(BaseModel):
    department : object
    region : object
    education: object
    gender: object
    age : int
    no_of_trainings : int
    previous_year_rating : float
    length_of_service: int
    kpis_met: int
    awards_won: int
    avg_training_score: int 
    

 
class Output(BaseModel):
    target : int

@app.post("/predict")
def predict(data: Input) -> Output:
    X_input = pd.DataFrame([[data.department,data.region,data.education,data.gender,
              data.age,data.no_of_trainings,data.previous_year_rating,data.length_of_service,data.kpis_met ,
              data.awards_won,data.avg_training_score]])
    
    X_input.columns = ['department','region','education','gender','age','no_of_trainings',
                       'previous_year_rating','length_of_service','kpis_met','awards_won','avg_training_score'
                       ]

    #load model
    model = joblib.load('promotion_pipeline_model.pkl')

    #predictmodel
    prediction = model.predict(X_input)
    
    #output
    return Output(target = prediction)
