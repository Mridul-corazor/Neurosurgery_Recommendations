from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_handler import QdrantStore
from typing import List, Optional
from nlp_services.sentiment_analysis import SentimeAnalysis
from nlp_services.emotions_analysis import EmotionsAnalysis
from nlp_services.behaviour_analysis import BehaviourAnalysis
from nlp_services.summarize import Summarizer
from recommendation import Recommendation
import logging
import json
logging.basicConfig(level=logging.INFO)
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    user_id:str
    k: Optional[int] = 2
    
class UpdateRequest(BaseModel):
    id: int
    new_text: str
    new_metadata: Optional[dict] = None

recommender = Recommendation()

@app.post("/update_text")
async def update_text(request: UpdateRequest):
    qdrant_handler = QdrantStore(collection_name="neurosurgery", url="http://localhost:6333")
    qdrant_handler.update_text(request.id, request.new_text, request.new_metadata)
    return {"message": "Text updated successfully"}

@app.post("/recommedation")
async def get_recommendation(request: QueryRequest):
    query = request.query
    user_id = request.user_id
    qdrant_handler = QdrantStore(collection_name="neurosurgery", url="http://localhost:6333")
    sentiment_analyzer = SentimeAnalysis()
    emotion_analyzer = EmotionsAnalysis()
    behaviour_analyzer = BehaviourAnalysis()
    
    # behaviour_analysis = behaviour_analyzer.analyze_gemini(query)
    behaviour_analysis = behaviour_analyzer.analyze(query,"llama-3.3-70b-versatile")
    behaviour_analysis = json.loads(behaviour_analysis)
    logging.info(f"Behaviour analysis: {behaviour_analysis}")
    summarizer = Summarizer()
    profile_summary = summarizer.analyze(query,"llama-3.3-70b-versatile")
    logging.info(f"Profile Summary: {profile_summary}")
    sentiment = sentiment_analyzer.analyze(behaviour_analysis['summary'])
    logging.info(f"Sentiment analysis: {sentiment}")
    emotion = emotion_analyzer.analyze(behaviour_analysis['summary'])
    logging.info(f"Emotional analysis: {emotion}")
    results = qdrant_handler.similarity_search(behaviour_analysis['summary'], k=2)

    recommendations = recommender.recommend(
        user_id,
        context_vars={
            "patient_profile": query,
            "retrieved_text": results[0].page_content if results else "",
            "sentiment_analysis": sentiment,
            "emotional_state": emotion,
            "behavioral_analysis": behaviour_analysis
        }
    )
    logging.info(f"Recommendation: {recommendations}")
    
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
