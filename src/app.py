from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import uvicorn

# Initialize the FastAPI application
app = FastAPI(
    title="eCommerce Recommendation Engine",
    description="Project Capstone: API for personalized product recommendations",
    version="1.0.0"
)

# Define global variables for the model and product list
model = None
unique_products = []

# Load the trained model and product list when the server starts


@app.on_event("startup")
def load_model():
    global model, unique_products

    models_dir = r"C:\Users\Nathan\OneDrive\Documents\AIM\Capstone Project\models"

    try:
        with open(os.path.join(models_dir, 'svd_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(models_dir, 'unique_products.pkl'), 'rb') as f:
            unique_products = pickle.load(f)
        print("Model successfully loaded into memory.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Model files not found. Ensure you exported them from the notebook.")

# Define the expected structure of incoming API requests


class RecommendationRequest(BaseModel):
    user_id: str
    top_n: int = 5

# Root endpoint for health checks


@app.get("/")
def read_root():
    return {"status": "Online", "message": "Welcome to the AI Recommendation API!"}

# The main prediction endpoint


@app.post("/recommend")
def get_recommendations(req: RecommendationRequest):
    if not model or not unique_products:
        raise HTTPException(
            status_code=500, detail="Machine Learning model is not loaded.")

    predictions = []

    # Predict the rating for every single product in the catalog for this specific user
    for product_id in unique_products:
        # model.predict returns a Prediction object; '.est' is the estimated star rating
        pred = model.predict(uid=req.user_id, iid=product_id)
        predictions.append((product_id, pred.est))

    # Sort the catalog by the highest predicted ratings
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Slice the top N products
    top_products = [
        {"product_id": pid, "predicted_rating": round(rating, 2)}
        for pid, rating in predictions[:req.top_n]
    ]

    return {
        "user_id": req.user_id,
        "recommendations": top_products
    }


# This allows you to run the script directly from the terminal
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
