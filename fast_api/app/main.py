from fastapi import FastAPI
from backend.app.routers import recommendations, users

# Create FastAPI instance
app = FastAPI(
    title="Game Recommendation API",
    description="An API for providing game recommendations based on collaborative and content-based filtering.",
    version="1.0.0"
)

# Include routers for different endpoints
app.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])
app.include_router(users.router, prefix="/users", tags=["Users"])

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Game Recommendation API!"}
