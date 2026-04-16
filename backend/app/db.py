import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

# ── LOCAL-READY CONFIG ────────────────────────────────────────────────────────
# Falls back to local MongoDB if environment variable is missing
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")

try:
    client = MongoClient(
        MONGO_URL,
        serverSelectionTimeoutMS=2000, # Faster timeout for local
        connectTimeoutMS=5000,
        socketTimeoutMS=30000,
        maxPoolSize=10,
        retryWrites=True,
    )
    # Verify connection on startup
    client.admin.command("ping")
    logger.info(f"✅ MongoDB connected successfully to: {MONGO_URL.split('@')[-1] if '@' in MONGO_URL else MONGO_URL}")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    logger.warning("⚠️ Make sure MongoDB is running locally or MONGO_URL is set.")
    # In a real local dev, you might want to mock the DB here, 
    # but usually, we just expect the user to have MongoDB installed.
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
