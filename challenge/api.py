import fastapi
# from fastapi import Request, Response
import uvicorn
# from model import DelayModel
from challenge.model import DelayModel
from typing import List, Dict
from fastapi import HTTPException
# from fastapi.responses import JSONResponse
import json
import pandas as pd
app = fastapi.FastAPI()
#load the model
model = DelayModel()
initial_features = {
    "OPERA_Latin American Wings": 1,
    "MES_7": 1,
    "MES_10": 1,
    "OPERA_Grupo LATAM": 1,
    "MES_12": 1,
    "TIPOVUELO_I": 1,
    "MES_4": 1,
    "MES_11": 1,
    "OPERA_Sky Airline": 1,
    "OPERA_Copa Air": 1
}
features_test = pd.DataFrame(initial_features,index=[0])




@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(flights: Dict) -> dict:
    try:
        flights = flights["flights"]
        print(flights)
        # Validar la estructura de los datos de vuelo
        if not all(isinstance(flight, dict) for flight in flights):
            raise ValueError("Invalid format for flight data")
        
        for flight in flights:
            
            if "MES" in flight and flight["MES"] > 12:
                raise ValueError("Invalid value for 'MES' field. Month must be between 1 and 12.")
        
            # Preprocesar los datos recibidos para hacer predicciones
            matrix = pd.DataFrame(flight,index=[0])
            features_test['OPERA_Latin American Wings'] = 1 if matrix['OPERA'].values[0] == 'Latin American Wings' else 0
            features_test['MES_7'] = 1 if matrix['MES'].values[0] == 7 else 0
            features_test['MES_10'] = 1 if matrix['MES'].values[0] == 10 else 0
            features_test['OPERA_Grupo LATAM'] = 1 if matrix['OPERA'].values[0] == 'Grupo LATAM' else 0
            features_test['MES_12'] = 1 if matrix['MES'].values[0] == 12 else 0
            features_test['TIPOVUELO_I'] = 1 if matrix['TIPOVUELO'].values[0] == 'I' else 0
            features_test['MES_4'] = 1 if matrix['MES'].values[0] == 4 else 0
            features_test['MES_11'] = 1 if matrix['MES'].values[0] == 11 else 0
            features_test['OPERA_Sky Airline'] = 1 if matrix['OPERA'].values[0] == 'Sky Airline' else 0
            features_test['OPERA_Copa Air'] = 1 if matrix['OPERA'].values[0] == 'Copa Air' else 0        
        # Hacer predicciones usando el modelo
        predictions = model.predict(features=features_test)
        
        # Devolver las predicciones
        return {"predict": predictions}
    except ValueError as e:
        # Capturar errores de validación y devolver un error 400 con mensaje personalizado
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(e)
        # Capturar cualquier otra excepción y devolver un error 400 con mensaje personalizado
        raise HTTPException(status_code=400, detail=f'Internal Server Error {e}')


if __name__=='__main__':
    uvicorn.run("api:app",port=8000,reload=True,host="localhost")