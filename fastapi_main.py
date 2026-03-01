import uvicorn
import os
import json
import logging
import redis
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ローカルモジュールのインポート
from api_character_loader import GetCharacterAttributes, CharacterAttributes
import audio_api_service
import text_api_service
import inference_emotion_detect

# ロギング設定（標準出力）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS設定：フロントエンドからのアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.1.36:5173",
        "http://192.168.1.41:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストボディ用モデル定義
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

# グローバル変数の定義
audio_gen_instance: audio_api_service.SimpleAudioGenerator = None
text_gen_instance: text_api_service.SimpleTextGenerator = None
character_configs: List[CharacterAttributes] =[]
emotion_detector = inference_emotion_detect.EmotionDetect()
emotion_model = None

# Redisクライアントの初期化（メッセージブローカーとして使用）
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化処理"""
    global audio_gen_instance, text_gen_instance, character_configs, emotion_model
    
    logger.info("FastAPI API Gatewayの起動処理を開始します...")

    # 1. 感情分析モデルの初期化（CPUで実行可能かつ軽量なためGatewayに配置）
    logger.info("感情分析モデルを初期化中...")
    emotion_model = emotion_detector.launch_emotion_detect()
    
    # 2. キャラクター設定の読み込み
    try:
        get_char_attrs = GetCharacterAttributes()
        character_configs = get_char_attrs.character_class_list
        if not character_configs:
            raise ValueError("キャラクター設定が読み込めませんでした。設定ファイルを確認してください。")
        logger.info(f"{len(character_configs)}体のキャラクター設定を読み込みました。")
    except Exception as e:
        logger.error(f"キャラクター設定の読み込みに失敗しました: {e}")
        raise RuntimeError(f"キャラクター設定の読み込みに失敗: {e}")

    # 3. APIキーの読み込み (JSONファイルから)
    gemini_keys = []
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

    # 4. 音声生成器（Gateway側）の初期化とRedisの登録
    audio_api_service.set_redis_client(redis_client)
    audio_gen_instance = audio_api_service.SimpleAudioGenerator(character_configs)
    await audio_gen_instance.load_gemini_api_key()
    await audio_gen_instance.load_azure_tts_subscription_key()
    audio_api_service.set_audio_generator_instance(audio_gen_instance)
    audio_api_service.set_emotion_model(emotion_model)

    # 5. テキスト生成器の初期化
    text_gen_instance = text_api_service.SimpleTextGenerator(character_configs, gemini_keys, modelscope_keys)
    
    logger.info("API Gatewayの起動が完了しました。")

# オーディオルーターの登録
app.include_router(audio_api_service.audio_router, prefix="/api/audio")

@app.get("/characters", response_model=List[CharacterInfo])
async def get_characters():
    """ロードされたキャラクター情報のリストを取得するエンドポイント"""
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
    """テキスト応答生成のエンドポイント（LLM呼び出し）"""
    if text_gen_instance is None:
        raise HTTPException(status_code=500, detail="テキスト生成器が初期化されていません。")

    try:
        response_text, speaker_char_index = await text_gen_instance.generate_text_response_for_api(
            request.user_message,
            request.character_index,
            request.chat_history,
            request.language_choice,
            request.sakiko_state,
            request.use_modelscope,
            request.is_dual_character_mode,
            request.secondary_character_index
        )
        return {"response_text": response_text, "speaker_char_index": speaker_char_index}
    except Exception as e:
        logger.error(f"/generate_text_response でエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=f"テキスト生成に失敗しました: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)