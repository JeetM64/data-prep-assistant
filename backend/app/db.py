import os
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise ValueError("MONGO_URL is not set")

client = MongoClient(MONGO_URL)

db = client["ml_analyzer"]

reports_collection = db["reports"]
users_collection = db["users"]