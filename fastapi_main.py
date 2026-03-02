import uvicorn
import os
import json
import logging
import redis
from dotenv import load_dotenv
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api_character_loader import GetCharacterAttributes, CharacterAttributes
import audio_api_service
import text_api_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

load_dotenv()
CORS_ORIGINS_STR = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://192.168.1.43:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS_STR.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_message: str
    character_index: int
    chat_history: List[Dict]
    language_choice: str
    sakiko_state: bool = True
    use_modelscope: bool = False
    is_dual_character_mode: bool = False
    secondary_character_index: Optional[int] = None

class CharacterInfo(BaseModel):
    id: int
    character_name: str
    icon_path: Optional[str]
    live2d_json: str
    character_description: str

audio_gen_instance: audio_api_service.SimpleAudioGenerator = None
text_gen_instance: text_api_service.SimpleTextGenerator = None
character_configs: List[CharacterAttributes] =[]

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

@app.on_event("startup")
async def startup_event():
    global audio_gen_instance, text_gen_instance, character_configs
    
    logger.info("FastAPI API Gatewayの起動処理を開始します...")

    try:
        get_char_attrs = GetCharacterAttributes()
        character_configs = get_char_attrs.character_class_list
        if not character_configs:
            raise ValueError("キャラクター設定が読み込めませんでした。設定ファイルを確認してください。")
        logger.info(f"{len(character_configs)}体のキャラクター設定を読み込みました。")
    except Exception as e:
        logger.error(f"キャラクター設定の読み込みに失敗しました: {e}")
        raise RuntimeError(f"キャラクター設定の読み込みに失敗: {e}")

    gemini_keys =[]
    modelscope_keys =[]
    api_key_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "api_keys.json"))
    
    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                gemini_keys = data.get("gemini",[])
                modelscope_keys = data.get("modelscope",[])
        except Exception as e:
            logger.error(f"APIキーの読み込みに失敗しました: {e}")
            
    if not gemini_keys:
        gemini_keys = ["YOUR_DEFAULT_GEMINI_API_KEY_HERE"]
    if not modelscope_keys:
        modelscope_keys = ["ms-default-token"]

    audio_api_service.set_redis_client(redis_client)
    audio_gen_instance = audio_api_service.SimpleAudioGenerator(character_configs)
    await audio_gen_instance.load_gemini_api_key()
    await audio_gen_instance.load_azure_tts_subscription_key()
    audio_api_service.set_audio_generator_instance(audio_gen_instance)

    text_gen_instance = text_api_service.SimpleTextGenerator(character_configs, gemini_keys, modelscope_keys)
    
    logger.info("API Gatewayの起動が完了しました。")

app.include_router(audio_api_service.audio_router, prefix="/api/audio")

@app.get("/characters", response_model=List[CharacterInfo])
async def get_characters():
    global character_configs
    return[
        CharacterInfo(
            id=i,
            character_name=char.character_name,
            icon_path=char.icon_path,
            live2d_json=char.live2d_json,
            character_description=char.character_description
        ) for i, char in enumerate(character_configs)
    ]

@app.post("/generate_text_response")
async def generate_text_response_endpoint(request: ChatRequest):
    if text_gen_instance is None:
        raise HTTPException(status_code=500, detail="テキスト生成器が初期化されていません。")

    try:
        response_text, speaker_char_index, emotion_tag = await text_gen_instance.generate_text_response_for_api(
            request.user_message,
            request.character_index,
            request.chat_history,
            request.language_choice,
            request.sakiko_state,
            request.use_modelscope,
            request.is_dual_character_mode,
            request.secondary_character_index
        )
        return {
            "response_text": response_text, 
            "speaker_char_index": speaker_char_index,
            "emotion": emotion_tag
        }
    except Exception as e:
        logger.error(f"/generate_text_response でエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=f"テキスト生成に失敗しました: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)