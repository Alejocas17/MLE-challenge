import fastapi
import uvicorn
app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    return


if __name__=='__main__':
    uvicorn.run("api:app",port=8000,reload=True,host="localhost")