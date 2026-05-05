from fastapi import FastAPI
from fastapi.responses import Response
import requests
import joblib
import pandas as pd
import base64

import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

app = FastAPI()

model = joblib.load("model/pokemon_model.pkl")


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = create_features(df)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    prompt = build_prompt(data)
    img, err = generate_image(prompt)

    is_img_null_flag = ""

    if img is not None:
        img_base64 = base64.b64encode(img).decode("utf-8")
        is_img_null_flag = "0"
    else:
        img_base64 = None
        is_img_null_flag = "1"

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "image": img_base64,
        "is_img_null": is_img_null_flag,
        "image_error": str(err) if err else None
    }

def save_image(img, filename="avatar.jpg"):
    with open(filename, "wb") as f:
        f.write(img)

def create_features(df):
    df = df.copy()
    
    df["offensive"] = df["attack"] + df["sp_attack"]
    df["defensive"] = df["defense"] + df["sp_defense"]
    
    stats = ["hp","attack","defense","sp_attack","sp_defense","speed"]
    df["balance"] = df[stats].std(axis=1)
    
    df["physical_ratio"] = df["attack"] / (df["sp_attack"] + 1)
    df["special_ratio"] = df["sp_attack"] / (df["attack"] + 1)
    
    df["atk_def_ratio"] = df["attack"] / (df["defense"] + 1)
    df["Sp_atk_sp_def_ratio"] = df["sp_attack"] / (df["sp_defense"] + 1)
    
    return df



headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

def build_prompt(d):
    # minimal prompt – sadece avatar üret
    return (
        "a pokemon-like creature avatar, anime style, "
        "clean background, centered, character art"
    )

def generate_image(prompt):
    try:
        r = requests.post(
            MODEL_URL,
            headers=headers,
            json={"inputs": prompt},  # HF JSON body alıyor, form-data değil
            timeout=60
        )

        content_type = r.headers.get("content-type", "")

        if r.status_code == 200 and "image" in content_type:
            with open("./avatar.jpeg", "wb") as f:
                f.write(r.content)
            return r.content, None
        else:
            return None, r.json()

    except Exception as e:
        return None, str(e)