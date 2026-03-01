import os
import uvicorn
from dotenv import load_dotenv

load_dotenv("/etc/secrets/.env")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print(f"🚀 Starting server on port {port}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
