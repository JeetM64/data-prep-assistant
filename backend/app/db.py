import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

logger = logging.getLogger(__name__)

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise ValueError("❌ MONGO_URL environment variable is not set. Add it in Render → Environment.")

try:
    client = MongoClient(
        MONGO_URL,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        socketTimeoutMS=30000,
        maxPoolSize=10,
        retryWrites=True,
    )
    # Verify connection on startup
    client.admin.command("ping")
    logger.info("✅ MongoDB connected successfully")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    raise

db = client["ml_analyzer"]

reports_collection  = db["reports"]
users_collection    = db["users"]

# Indexes (idempotent — safe to run every startup)
try:
    users_collection.create_index("email", unique=True)
    reports_collection.create_index([("user_id", 1), ("created_at", -1)])
    reports_collection.create_index("created_at")
    logger.info("✅ MongoDB indexes ensured")
except Exception as e:
    logger.warning(f"Index creation warning (non-fatal): {e}")