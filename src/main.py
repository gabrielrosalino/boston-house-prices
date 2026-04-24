from fastapi import FastAPI

app = FastAPI()

@app.get('/predict/{status}')
async def home(status : float):
    ANGULAR_COEFF = -0.9520197493796366
    LINEAR_COEFF = 34.80760366293856

    predict = ANGULAR_COEFF * status + LINEAR_COEFF
    return {"value": predict}