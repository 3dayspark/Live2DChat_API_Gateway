import base64
import re
import os
import json
import time
import shutil
import struct
import logging
import asyncio
import uuid
from typing import List, Dict, Optional, Any
from xml.etree import ElementTree

# 環境変数の設定: HuggingFaceとTransformersのオフラインモードを強制
# オンライン接続によるGSVの遅延（ラグ）を防ぐための設定
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import soundfile as sf
import numpy as np
import requests
import edge_tts

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# キャラクター設定のインポート
from api_character_loader import CharacterAttributes

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# グローバル変数：感情分析モデルおよびRedisクライアント
_global_emotion_model: Any = None
_redis_client: Any = None

# 外部サービスURL設定
GEMINI_TTS_URL = "https://asynchronousblocking.asia/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
RVC_SERVICE_URL = "http://127.0.0.1:8001/rvc_convert"  # RVC FastAPIサービスのポート設定

# Azure TTS 関連設定
AZURE_TTS_REGION = "japanwest"
AZURE_TTS_ENDPOINT = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
AZURE_TTS_TOKEN_URL = f"https://{AZURE_TTS_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
AZURE_TTS_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
AZURE_TTS_OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"
AZURE_TTS_USER_AGENT = "Chat_backend"


def parse_audio_mime_type(mime_type: str) -> dict[str, Optional[int]]:
    """MIMEタイプから音声パラメータを解析するヘルパー関数"""
    bits_per_sample = 16
    rate = 24000
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """PCMデータをWAVフォーマットに変換する"""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (PCM)
        1,                # AudioFormat (PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size
    )
    return header + audio_data


class SimpleAudioGenerator:
    """音声生成ロジックとタスクディスパッチを管理するクラス"""

    def __init__(self, character_list: List[CharacterAttributes]):
        self.character_list = character_list
        
        # 絶対パスの構築
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 日本語合成時の読み間違い（人名など）を修正する辞書
        self.replacements_jap = {
            '豊川祥子': 'とがわさきこ',
            '祥子': 'さきこ',
            '三角初華': 'みすみういか',
            '初華': 'ういか',
            '若葉睦': 'わかばむつみ',
            '睦': 'むつみ',
            '八幡海鈴': 'やはたうみり',
            '海鈴': 'うみり',
            '海铃': 'うみり',
            '祐天寺': 'ゆうてんじ',
            '若麦': 'にゃむ',
            '喵梦': 'にゃむ',
            '高松燈':'たかまつともり',
            '燈':'ともり',
            '灯': 'ともり',
            '椎名立希':'しいなたき',
            '莉莎':'リサ',
            '愛音':'アノン',
            '素世': 'そよ',
            '爽世': 'そよ',
            '千早愛音':'ちはやアノン',
            '爱音': 'アノン',
            '要楽奈':'かなめらーな',
            '楽奈': 'らーな',
            '春日影':'はるひかげ',
            'Doloris':'ドロリス',
            'Mortis':'モーティス',
            'Timoris':'ティモリス',
            'Amoris':'アモーリス',
            'Oblivionis':'オブリビオニス',
            'live':'ライブ',
            'RiNG':'リング',
            '珠手知由':'たまでちゆ',
            'CHUCHU':'チュチュ',
            'CHU²':'チュチュ',
            'CHU2':'チュチュ',
            '友希那':'ゆきな',
            '纱夜':'サヨ',
            '牛肉干':'ジャーキー',
            'Roselia':'ロゼリア',
            '垃圾桶':'ゴミ箱',
            '髪型':'かみがた',
            'RAISE A SUILEN':'レイズアスイレン',
            'Senior Yukina':'ゆきな先輩',
            'MyGO!!!!!':'まいご'
        }
        
        # 中国語合成時の読み間違いを修正する辞書
        self.replacements_chi ={
            'CRYCHIC':'C团',
            'live':"演出",
            'RiNG':"ring",
            'Doloris': '初华',
            'Mortis': '睦',
            'Timoris': '海铃',
            'Amoris': '喵梦',
            'Oblivionis': '我',
            'MyGO':'mygo',
            'ちゃん':'',
            'CHU²':'楚楚',
            'CHU2':'楚楚'
        }
        
        # 広東語合成時の読み間違いを修正する辞書
        self.replacements_yue = {
            '丰川祥子': 'fung1 cyun1 coeng4 zi2',
            '祥子': 'coeng4 zi2',
            '睦': 'muk6',
            '爱音': 'oi3 jam1',
            '千早爱音': 'cin1 zou2 oi3 jam1',
            '立希': 'lap6 hei1',
            '椎名立希': 'ceoi1 naa4 lap6 hei1',
            '爽世': 'song2 sai3',
            '要乐奈': 'jiu3 lok6 naa4',
            '乐奈': 'lok6 naa4',
            '春日影':'har1 jat6 jing2',
            'CRYCHIC': 'klai4 sik1',
            'Ave Mujica': 'aai1 wai1 muk6 zi1 gaa3',
            'Doloris': 'do1 lo4 lei6 si1',
            'Mortis': 'muk6 ti4 si1',
            'Timoris': 'ti4 mo1 lei6 si1',
            'Amoris': 'aa3 mo1 lei6 si1',
            'Oblivionis': 'o1 bi1 lip6 bi1 o1 nis1',
            'live': 'laai1 fuk6',
            'MyGO': 'mai5 go1',
            'RiNG': 'ring1',
        }

        self.GEMINI_API_KEY = None
        self.generated_audio_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_audio", "generated_audios_temp")
        os.makedirs(self.generated_audio_folder, exist_ok=True)

        # Edge TTS 設定
        self.EDGE_TTS_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
        self.EDGE_TTS_DEFAULT_RATE = "+0%"
        self.EDGE_TTS_DEFAULT_VOLUME = "+0%"

        # Azure TTS 設定
        self.AZURE_TTS_SUBSCRIPTION_KEY = None
        self.azure_tts_access_token = None
        self.azure_tts_token_expiry_time = 0

        self.last_gemini_request_time = 0
        self.gemini_request_interval = 2  # レート制限防止用のインターバル（秒）
        self.edge_tts_lock = asyncio.Lock()

    async def load_gemini_api_key(self):
        """API Key.txt から Gemini API キーを読み込む"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        key_file_path = os.path.join(script_dir, "..", "API Key.txt")
        try:
            with open(key_file_path, "r", encoding="utf-8") as f:
                self.GEMINI_API_KEY = f.read().strip()
            logger.info("Gemini API Keyの読み込みに成功しました。")
        except FileNotFoundError:
            logger.error(f"エラー: API Keyファイルが {key_file_path} に見つかりません。Gemini TTSは機能しません。")
            self.GEMINI_API_KEY = None
        except Exception as e:
            logger.error(f"Gemini API Keyの読み込み中にエラーが発生しました: {e}")
            self.GEMINI_API_KEY = None

    async def load_azure_tts_subscription_key(self):
        """API Key_Azure.txt から Azure TTS キーを読み込む"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        azure_key_file_path = os.path.join(script_dir, "..", "API Key_Azure.txt")
        logger.info(f"Azure TTSキーの読み込みを試行します: {azure_key_file_path}")
        try:
            with open(azure_key_file_path, "r", encoding="utf-8") as f:
                azure_key = f.read().strip()
                if azure_key:
                    self.AZURE_TTS_SUBSCRIPTION_KEY = azure_key
                    logger.info("Azure TTS Subscription Keyの読み込みに成功しました。")
                else:
                    logger.warning(f"'{azure_key_file_path}' は空です。Azure TTSは機能しません。")
        except FileNotFoundError:
            logger.error(f"エラー: Azure TTS Keyファイルが {azure_key_file_path} に見つかりません。Azure TTSは機能しません。")
            self.AZURE_TTS_SUBSCRIPTION_KEY = None
        except Exception as e:
            logger.error(f"Azure TTS Subscription Keyの読み込み中にエラーが発生しました: {e}")
            self.AZURE_TTS_SUBSCRIPTION_KEY = None

    async def _get_azure_tts_access_token(self):
        """Azure TTSのアクセストークンを取得または更新する"""
        if not self.AZURE_TTS_SUBSCRIPTION_KEY:
            raise RuntimeError("Azure TTS Subscription Keyがロードされていません。")

        if self.azure_tts_access_token and not self._is_azure_tts_token_expired():
            return self.azure_tts_access_token

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.AZURE_TTS_SUBSCRIPTION_KEY
            }
            response = requests.post(AZURE_TTS_TOKEN_URL, headers=headers, timeout=10)
            response.raise_for_status()
            self.azure_tts_access_token = str(response.text)
            self.azure_tts_token_expiry_time = time.time() + 9 * 60  # 有効期限（通常10分）より早めに更新
            logger.info("Azure TTS Access Tokenの取得に成功しました。")
            return self.azure_tts_access_token
        except requests.exceptions.RequestException as e:
            logger.error(f"Azure TTS Access Tokenの取得でエラーが発生しました: {e}")
            if hasattr(response, 'text'):
                logger.error(f"Azure TTS Token エラーレスポンス: {response.text}")
            raise RuntimeError(f"Azure TTS Access Tokenの取得に失敗しました: {e}")

    def _is_azure_tts_token_expired(self):
        """トークンの有効期限を確認する"""
        return not self.azure_tts_access_token or time.time() >= self.azure_tts_token_expiry_time

    def _cleanup_generated_audios_temp(self):
        """一時音声フォルダのクリーンアップを行う（使用後）"""
        logger.info(f"一時音声ファイルのクリーンアップを実行します: {self.generated_audio_folder}")
        try:
            for filename in os.listdir(self.generated_audio_folder):
                if filename.endswith(".wav") or filename.endswith(".txt"):
                    file_path = os.path.join(self.generated_audio_folder, filename)
                    try:
                        os.remove(file_path)
                        logger.debug(f"一時ファイルを削除しました: {filename}")
                    except OSError as e:
                        logger.warning(f"一時ファイル {filename} の削除でエラーが発生しました: {e}")
        except Exception as e:
            logger.error(f"generated_audios_temp クリーンアップ中のエラー: {e}")

    async def _generate_audio_gpt_sovits(
        self,
        text: str,
        character_index: int,
        audio_language_choice: str,
        sakiko_state: bool
    ) -> tuple[str, str]:
        """
        GPT-SoVITSを使用した音声合成処理。
        ※Redisキューを介してGPU Workerにタスクを非同期ディスパッチします。
        """
        global _redis_client, _global_emotion_model

        if _redis_client is None:
            raise RuntimeError("Redisクライアントが初期化されていません。")

        if not (0 <= character_index < len(self.character_list)):
            raise ValueError(f"無効なキャラクターインデックスです: {character_index}")

        current_char = self.character_list[character_index]

        # 1. テキスト前処理
        translation_pattern_to_remove = r"(?:\[翻译\]|\[翻訳\]).*?(?:\[翻译结束\]|\[翻訳結束\]|\[翻訳終了\])"
        cleaned_text = re.sub(translation_pattern_to_remove, "", text, flags=re.DOTALL).strip()

        processed_text = cleaned_text
        if audio_language_choice == '日英混合':
            processed_text = re.sub(r'CRYCHIC', 'クライシック', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'\bave\s*mujica\b', 'あヴぇムジカ', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'立希', ('りっき' if self.character_list[character_index].character_name == '爱音' else 'たき'), processed_text, flags=re.IGNORECASE)
            for key, value in self.replacements_jap.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        elif audio_language_choice == '粤英混合':
            for key, value in self.replacements_yue.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        else:  # 中国語
            for key, value in self.replacements_chi.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)

        pattern = r'^[^A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]+\'\''  # 文頭の記号除去
        processed_text = re.sub(pattern, '', processed_text)
        processed_text = processed_text.replace(' ', '')

        if processed_text == '' or processed_text == '不能送去合成':
            processed_text = '今年'  # エラー回避用のデフォルト値

        # 2. 感情分析の実行
        emotion_label = "neutral"
        if _global_emotion_model and processed_text.strip():
            try:
                emotion_result = await run_in_threadpool(_global_emotion_model, processed_text)
                emotion_label = emotion_result[0]['label']
                logger.debug(f"テキスト '{processed_text}' の感情分析結果: {emotion_label}")
            except Exception as e:
                logger.warning(f"感情分析に失敗しました: {e}")

        # 3. GPU Workerへのタスクペイロード作成
        task_id = str(uuid.uuid4())
        task_payload = {
            "task_id": task_id,
            "text": processed_text,
            "character_folder": current_char.character_folder_name, # フォルダ名だけを送る
            "audio_language_choice": audio_language_choice,
            "sakiko_state": sakiko_state,
        }

        # 4. キャラクター専用のRedisキューにタスクをプッシュ (Zero Cold Start戦略)
        queue_name = f"queue:audio:{current_char.character_folder_name}"
        try:
            _redis_client.rpush(queue_name, json.dumps(task_payload))
            logger.info(f"タスク {task_id} をキュー {queue_name} にプッシュしました。")
        except Exception as e:
            logger.error(f"Redisへのタスクプッシュに失敗しました: {e}")
            raise RuntimeError(f"タスクのディスパッチに失敗しました: {e}")

        # 5. 非同期ポーリングによる結果待機 (最大60秒)
        result_key = f"result:{task_id}"
        audio_base64 = None
        
        # 600回 * 0.1秒 = 60秒のタイムアウト設定
        for _ in range(600):
            result_data = _redis_client.get(result_key)
            if result_data:
                res = json.loads(result_data)
                
                if "error" in res:
                    logger.error(f"Workerでエラーが発生しました: {res['error']}")
                    raise RuntimeError(f"GPU Workerでの合成に失敗しました: {res['error']}")
                
                audio_base64 = res.get("audio_base64")
                
                # メモリリークを防ぐため、結果取得後にキーを削除
                _redis_client.delete(result_key)
                break
            
            # 0.1秒待機（イベントループをブロックしない）
            await asyncio.sleep(0.1)

        if not audio_base64:
            raise TimeoutError("GPU Workerからの応答がタイムアウトしました。Workerプロセスが稼働しているか確認してください。")

        return audio_base64, emotion_label

    async def _generate_audio_tts_rvc(self, text: str, character_index: int, audio_language_choice: str) -> tuple[str, str]:
        """Gemini TTS + RVC を使用した音声合成処理"""
        # 1. RVCモデルパスの取得
        if not (0 <= character_index < len(self.character_list)):
            raise ValueError(f"無効なキャラクターインデックスです: {character_index}")

        current_char = self.character_list[character_index]
        rvc_model_dir_id = current_char.rvc_model_dir_id
        rvc_index_dir_id = current_char.rvc_index_dir_id

        if not rvc_model_dir_id or not rvc_index_dir_id:
            raise ValueError(f"キャラクター '{current_char.character_name}' のRVCモデル/インデックスディレクトリIDが設定されていません。")

        # 2. Gemini TTS呼び出し
        if not self.GEMINI_API_KEY:
            raise RuntimeError("Gemini API Keyが読み込まれていません。TTS+RVCメソッドを使用できません。")

        # テキスト前処理
        translation_pattern_to_remove = r"(?:\[翻译\]|\[翻訳\]).*?(?:\[翻译结束\]|\[翻訳結束\]|\[翻訳終了\])"
        cleaned_text = re.sub(translation_pattern_to_remove, "", text, flags=re.DOTALL).strip()

        processed_text = cleaned_text
        if audio_language_choice == '日英混合':
            processed_text = re.sub(r'CRYCHIC', 'クライシック', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'\bave\s*mujica\b', 'あヴぇムジカ', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'立希', ('りっき' if self.character_list[character_index].character_name == '爱音' else 'たき'), processed_text, flags=re.IGNORECASE)
            for key, value in self.replacements_jap.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        elif audio_language_choice == '粤英混合':
            for key, value in self.replacements_yue.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        else:  # 中国語
            for key, value in self.replacements_chi.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)

        pattern = r'^[^A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]+\'\''
        processed_text = re.sub(pattern, '', processed_text)
        processed_text = processed_text.replace(' ', '')

        tts_payload = {
            "contents": [{
                "parts":[{"text": processed_text}]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Aoede"
                        }
                    }
                }
            },
            "model": "gemini-2.5-flash-preview-tts",
        }
        tts_headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.GEMINI_API_KEY,
        }

        emotion_label = "neutral"

        try:
            # 感情分析
            if _global_emotion_model and processed_text.strip():
                try:
                    emotion_result = await run_in_threadpool(_global_emotion_model, processed_text)
                    emotion_label = emotion_result[0]['label']
                    logger.debug(f"テキスト '{processed_text}' の感情分析結果 (TTS+RVC): {emotion_label}")
                except Exception as e:
                    logger.warning(f"TTS+RVC の感情分析に失敗しました: {e}")

            logger.info(f"Gemini TTSリクエストを送信中... ペイロード: {json.dumps(tts_payload, indent=2)}")
            response = requests.post(GEMINI_TTS_URL, headers=tts_headers, json=tts_payload, timeout=30)
            
            logger.info(f"Gemini TTSレスポンスステータス: {response.status_code}")
            response.raise_for_status()
            response_json = response.json()

            if "error" in response_json:
                error_message = response_json["error"].get("message", "Unknown Gemini TTS API error")
                logger.error(f"Gemini TTS API エラー: {error_message}")
                raise RuntimeError(f"Gemini TTS API エラー: {error_message}")

            base64_audio_data = response_json.get("candidates", [{}])[0]\
                                             .get("content", {})\
                                             .get("parts", [{}])[0]\
                                             .get("inlineData", {})\
                                             .get("data")

            if not base64_audio_data:
                raise RuntimeError("Gemini TTSからの音声データが空です (data field missing or empty).")

            decoded_audio_data = base64.b64decode(base64_audio_data)
            tts_wav_data = convert_to_wav(decoded_audio_data, "audio/L16;rate=24000")

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Gemini TTSからのHTTPエラー: {http_err}")
            raise RuntimeError(f"Gemini TTS HTTPエラー: {http_err}") from http_err
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini TTS呼び出し時のネットワークエラー: {e}")
            raise RuntimeError(f"Gemini TTSネットワークエラー: {e}") from e
        except Exception as e:
            logger.error(f"Gemini TTS呼び出し中の予期せぬエラー: {e}")
            raise RuntimeError(f"Gemini TTS予期せぬエラー: {e}") from e

        # 3. RVCサービスによる音質変換
        rvc_payload = {
            "audio_data_base64": base64.b64encode(tts_wav_data).decode('utf-8'),
            "rvc_model_relative_dir": rvc_model_dir_id, 
            "rvc_index_relative_dir": rvc_index_dir_id, 
            "pitch_shift": 0,
            "f0_method": "rmvpe",
            "index_rate": 0.75,
            "filter_radius": 3,
            "resample_sr": 0,
            "rms_mix_rate": 0.25,
            "protect": 0.33
        }
        rvc_headers = {"Content-Type": "application/json"}

        try:
            rvc_response = requests.post(RVC_SERVICE_URL, headers=rvc_headers, json=rvc_payload, timeout=60)
            rvc_response.raise_for_status()
            rvc_response_json = rvc_response.json()
            converted_audio_base64 = rvc_response_json.get("converted_audio_base64")

            if not converted_audio_base64:
                raise RuntimeError("RVCサービスからの変換後音声が空です。")

            converted_audio_bytes = base64.b64decode(converted_audio_base64)
            return base64.b64encode(converted_audio_bytes).decode("utf-8"), emotion_label

        except requests.exceptions.RequestException as e:
            logger.error(f"キャラクター {current_char.character_name} のRVCサービス呼び出しエラー: {e}")
            if hasattr(rvc_response, 'content'):
                logger.error(f"RVC エラーレスポンス: {rvc_response.text}")
            raise
        except Exception as e:
            logger.error(f"キャラクター {current_char.character_name} のRVC変換中に予期せぬエラーが発生しました: {e}")
            raise

    async def _generate_audio_edge_tts_rvc(self, text: str, character_index: int, audio_language_choice: str) -> tuple[str, str]:
        """Edge TTS + RVC を使用した音声合成処理"""
        # 1. RVCモデルパスの取得
        if not (0 <= character_index < len(self.character_list)):
            raise ValueError(f"無効なキャラクターインデックスです: {character_index}")

        current_char = self.character_list[character_index]
        rvc_model_dir_id = current_char.rvc_model_dir_id
        rvc_index_dir_id = current_char.rvc_index_dir_id

        if not rvc_model_dir_id or not rvc_index_dir_id:
            raise ValueError(f"キャラクター '{current_char.character_name}' のRVCモデル/インデックスディレクトリIDが設定されていません。")

        # 2. Edge TTS 音声生成
        edge_tts_voice = self.EDGE_TTS_DEFAULT_VOICE

        # カスタム音声設定の読み込み (azure_voice.txt)
        character_audio_base_dir_in_chat_backend = os.path.dirname(current_char.gptsovits_ref_audio)
        azure_voice_config_path = os.path.join(character_audio_base_dir_in_chat_backend, "azure_voice.txt")
        custom_voices =[]

        if os.path.exists(azure_voice_config_path):
            try:
                with open(azure_voice_config_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    custom_voices = [line.strip() for line in lines if line.strip()]
                logger.info(f"{azure_voice_config_path} からカスタム音声を読み込みました: {custom_voices}")
            except Exception as e:
                logger.warning(f"{azure_voice_config_path} からのカスタム音声の読み込みに失敗しました: {e}")

        if custom_voices:
            if audio_language_choice == "日英混合":
                if len(custom_voices) >= 2:
                    edge_tts_voice = custom_voices[1]
                    logger.info(f"日英混合用にカスタムEdge TTS音声を使用します: {edge_tts_voice}")
                else:
                    logger.warning(f"'{azure_voice_config_path}' に日英混合用の設定がありません。デフォルトを使用します。")
            elif audio_language_choice == "粤英混合":
                if len(custom_voices) >= 3:
                    edge_tts_voice = custom_voices[2]
                    logger.info(f"粤英混合用にカスタムEdge TTS音声を使用します: {edge_tts_voice}")
                else:
                    logger.warning(f"'{azure_voice_config_path}' に粤英混合用の設定がありません。デフォルトを使用します。")
            else:  # デフォルト/中英混合
                if len(custom_voices) >= 1:
                    edge_tts_voice = custom_voices[0]
                    logger.info(f"デフォルト/中英混合用にカスタムEdge TTS音声を使用します: {edge_tts_voice}")
                else:
                    logger.warning(f"'{azure_voice_config_path}' が空か設定が不正です。デフォルトを使用します。")

        if audio_language_choice == "日英混合":
            if not custom_voices or len(custom_voices) < 2:
                edge_tts_voice = "ja-JP-AoiNeural"
        elif audio_language_choice == "粤英混合":
            if not custom_voices or len(custom_voices) < 3:
                edge_tts_voice = "zh-HK-HiuGaaiNeural"

        # テキスト前処理
        translation_pattern_to_remove = r"(?:\[翻译\]|\[翻訳\]).*?(?:\[翻译结束\]|\[翻訳結束\]|\[翻訳終了\])"
        cleaned_text = re.sub(translation_pattern_to_remove, "", text, flags=re.DOTALL).strip()

        processed_text = cleaned_text
        if audio_language_choice == '日英混合':
            processed_text = re.sub(r'CRYCHIC', 'クライシック', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'\bave\s*mujica\b', 'あヴぇムジカ', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'立希', ('りっき' if self.character_list[character_index].character_name == '爱音' else 'たき'), processed_text, flags=re.IGNORECASE)
            for key, value in self.replacements_jap.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        elif audio_language_choice == '粤英混合':
            for key, value in self.replacements_yue.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        else:  # 中国語
            for key, value in self.replacements_chi.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)

        pattern = r'^[^A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]+\'\''
        processed_text = re.sub(pattern, '', processed_text)
        processed_text = processed_text.replace(' ', '')

        emotion_label = "neutral"

        try:
            # 感情分析
            global _global_emotion_model
            if _global_emotion_model and processed_text.strip():
                try:
                    emotion_result = await run_in_threadpool(_global_emotion_model, processed_text)
                    emotion_label = emotion_result[0]['label']
                    logger.debug(f"テキスト '{processed_text}' の感情分析結果 (EdgeTTS+RVC): {emotion_label}")
                except Exception as e:
                    logger.warning(f"EdgeTTS+RVC の感情分析に失敗しました: {e}")

            communicate = edge_tts.Communicate(
                processed_text,
                edge_tts_voice,
                rate=self.EDGE_TTS_DEFAULT_RATE,
                volume=self.EDGE_TTS_DEFAULT_VOLUME,
            )

            audio_data_chunks =[]
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data_chunks.append(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
            
            edge_tts_raw_audio_bytes = b"".join(audio_data_chunks)
            edge_tts_wav_data = edge_tts_raw_audio_bytes
            logger.info(f"Edge TTSによる音声生成が完了しました。サイズ: {len(edge_tts_wav_data)} バイト")

        except Exception as e:
            logger.error(f"キャラクター {current_char.character_name} のEdge TTS呼び出しエラー: {e}")
            raise RuntimeError(f"Edge TTS呼び出しエラー: {e}")

        # 3. RVCサービスによる音質変換
        rvc_payload = {
            "audio_data_base64": base64.b64encode(edge_tts_wav_data).decode('utf-8'),
            "rvc_model_relative_dir": rvc_model_dir_id, 
            "rvc_index_relative_dir": rvc_index_dir_id,
            "pitch_shift": 0,
            "f0_method": "rmvpe",
            "index_rate": 0.75,
            "filter_radius": 3,
            "resample_sr": 0,
            "rms_mix_rate": 0.25,
            "protect": 0.33
        }
        rvc_headers = {"Content-Type": "application/json"}

        try:
            rvc_response = requests.post(RVC_SERVICE_URL, headers=rvc_headers, json=rvc_payload, timeout=60)
            rvc_response.raise_for_status()
            rvc_response_json = rvc_response.json()
            converted_audio_base64 = rvc_response_json.get("converted_audio_base64")

            if not converted_audio_base64:
                raise RuntimeError("RVCサービスからの変換後音声が空です。")

            converted_audio_bytes = base64.b64decode(converted_audio_base64)
            return base64.b64encode(converted_audio_bytes).decode("utf-8"), emotion_label

        except requests.exceptions.RequestException as e:
            logger.error(f"キャラクター {current_char.character_name} のRVCサービス呼び出しエラー: {e}")
            if hasattr(rvc_response, 'content'):
                logger.error(f"RVC エラーレスポンス: {rvc_response.text}")
            raise
        except Exception as e:
            logger.error(f"キャラクター {current_char.character_name} のRVC変換中に予期せぬエラーが発生しました: {e}")
            raise

    async def _generate_audio_azure_tts_rvc(self, text: str, character_index: int, audio_language_choice: str) -> tuple[str, str]:
        """Azure TTS + RVC を使用した音声合成処理"""
        # 1. RVCモデルパスの取得
        if not (0 <= character_index < len(self.character_list)):
            raise ValueError(f"無効なキャラクターインデックスです: {character_index}")

        current_char = self.character_list[character_index]
        rvc_model_dir_id = current_char.rvc_model_dir_id
        rvc_index_dir_id = current_char.rvc_index_dir_id

        if not rvc_model_dir_id or not rvc_index_dir_id:
            raise ValueError(f"キャラクター '{current_char.character_name}' のRVCモデル/インデックスディレクトリIDが設定されていません。")

        # 2. Azure TTS 音声生成
        if not self.AZURE_TTS_SUBSCRIPTION_KEY:
            raise RuntimeError("Azure TTS Subscription Keyが読み込まれていません。Azure TTS+RVCメソッドを使用できません。")

        # テキスト前処理
        translation_pattern_to_remove = r"(?:\[翻译\]|\[翻訳\]).*?(?:\[翻译结束\]|\[翻訳結束\]|\[翻訳終了\])"
        cleaned_text = re.sub(translation_pattern_to_remove, "", text, flags=re.DOTALL).strip()

        processed_text = cleaned_text
        if audio_language_choice == '日英混合':
            processed_text = re.sub(r'CRYCHIC', 'クライシック', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'\bave\s*mujica\b', 'あヴぇムジカ', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'立希', ('りっき' if self.character_list[character_index].character_name == '爱音' else 'たき'), processed_text, flags=re.IGNORECASE)
            for key, value in self.replacements_jap.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        elif audio_language_choice == '粤英混合':
            for key, value in self.replacements_yue.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)
        else:  # 中国語
            for key, value in self.replacements_chi.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)

        pattern = r'^[^A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]+\'\''
        processed_text = re.sub(pattern, '', processed_text)
        processed_text = processed_text.replace(' ', '')

        emotion_label = "neutral"

        try:
            # 感情分析
            global _global_emotion_model
            if _global_emotion_model and processed_text.strip():
                try:
                    emotion_result = await run_in_threadpool(_global_emotion_model, processed_text)
                    emotion_label = emotion_result[0]['label']
                    logger.debug(f"テキスト '{processed_text}' の感情分析結果 (AzureTTS+RVC): {emotion_label}")
                except Exception as e:
                    logger.warning(f"AzureTTS+RVC の感情分析に失敗しました: {e}")

            access_token = await self._get_azure_tts_access_token()

            # SSMLペイロードの構築
            azure_tts_voice_name = AZURE_TTS_DEFAULT_VOICE

            # カスタム音声設定の読み込み
            character_audio_base_dir_in_chat_backend = os.path.dirname(current_char.gptsovits_ref_audio)
            azure_voice_config_path = os.path.join(character_audio_base_dir_in_chat_backend, "azure_voice.txt")
            custom_voices =[]

            if os.path.exists(azure_voice_config_path):
                try:
                    with open(azure_voice_config_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        custom_voices =[line.strip() for line in lines if line.strip()]
                    logger.info(f"{azure_voice_config_path} からカスタム音声を読み込みました: {custom_voices}")
                except Exception as e:
                    logger.warning(f"{azure_voice_config_path} からのカスタム音声の読み込みに失敗しました: {e}")

            if custom_voices:
                if audio_language_choice == "日英混合":
                    if len(custom_voices) >= 2:
                        azure_tts_voice_name = custom_voices[1]
                        logger.info(f"日英混合用にカスタムAzure TTS音声を使用します: {azure_tts_voice_name}")
                    else:
                        logger.warning(f"'{azure_voice_config_path}' に日英混合用の設定がありません。デフォルトを使用します。")
                elif audio_language_choice == "粤英混合":
                    if len(custom_voices) >= 3:
                        azure_tts_voice_name = custom_voices[2]
                        logger.info(f"粤英混合用にカスタムAzure TTS音声を使用します: {azure_tts_voice_name}")
                    else:
                        logger.warning(f"'{azure_voice_config_path}' に粤英混合用の設定がありません。デフォルトを使用します。")
                else:  # デフォルト/中英混合
                    if len(custom_voices) >= 1:
                        azure_tts_voice_name = custom_voices[0]
                        logger.info(f"デフォルト/中英混合用にカスタムAzure TTS音声を使用します: {azure_tts_voice_name}")
                    else:
                        logger.warning(f"'{azure_voice_config_path}' が空か設定が不正です。デフォルトを使用します。")

            xml_body = ElementTree.Element('speak', version='1.0')
            voice = ElementTree.SubElement(xml_body, 'voice')

            if audio_language_choice == "日英混合":
                if not custom_voices or len(custom_voices) < 2:
                    azure_tts_voice_name = "ja-JP-AoiNeural"
                xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', 'ja-JP')
                voice.set('{http://www.w3.org/XML/1998/namespace}lang', 'ja-JP')
            elif audio_language_choice == "粤英混合":
                if not custom_voices or len(custom_voices) < 3:
                    azure_tts_voice_name = "zh-HK-HiuGaaiNeural"
                xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', 'zh-HK')
                voice.set('{http://www.w3.org/XML/1998/namespace}lang', 'zh-HK')
            else:
                xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', 'zh-CN')
                voice.set('{http://www.w3.org/XML/1998/namespace}lang', 'zh-CN')

            voice.set('name', azure_tts_voice_name)
            voice.text = processed_text
            body = ElementTree.tostring(xml_body, encoding='utf-8')

            azure_tts_headers = {
                'Authorization': 'Bearer ' + access_token,
                'Content-Type': 'application/ssml+xml',
                'X-Microsoft-OutputFormat': AZURE_TTS_OUTPUT_FORMAT,
                'User-Agent': AZURE_TTS_USER_AGENT
            }

            response = requests.post(AZURE_TTS_ENDPOINT, headers=azure_tts_headers, data=body, timeout=30)
            response.raise_for_status()
            azure_tts_audio_data = response.content

        except requests.exceptions.RequestException as e:
            logger.error(f"キャラクター {current_char.character_name} のAzure TTS呼び出しエラー: {e}")
            if hasattr(response, 'content'):
                logger.error(f"Azure TTS エラーレスポンス: {response.content.decode('utf-8', errors='ignore')}")
            raise RuntimeError(f"Azure TTS呼び出しエラー: {e}")
        except Exception as e:
            logger.error(f"キャラクター {current_char.character_name} のAzure TTS実行中に予期せぬエラーが発生しました: {e}")
            raise RuntimeError(f"Azure TTS予期せぬエラー: {e}")

        # 3. RVCサービスによる音質変換
        base64_azure_tts_audio = base64.b64encode(azure_tts_audio_data).decode('utf-8')

        rvc_payload = {
            "audio_data_base64": base64_azure_tts_audio,
            "rvc_model_relative_dir": rvc_model_dir_id, 
            "rvc_index_relative_dir": rvc_index_dir_id,
            "f0_up_key": 0,
            "f0_method": "rmvpe",
            "protect": 0.5,
            "index_rate": 0.75,
            "resample_sr": 0,
            "rms_mix_rate": 1,
            "tuner_steps": 200
        }
        rvc_headers = {"Content-Type": "application/json"}

        logger.debug(f"キャラクター {current_char.character_name} のRVCリクエストを送信中...")

        try:
            rvc_response = requests.post(RVC_SERVICE_URL, headers=rvc_headers, json=rvc_payload, timeout=60)
            rvc_response.raise_for_status()
            rvc_response_json = rvc_response.json()

            logger.debug(f"RVCレスポンスを受信しました: {rvc_response_json}")

            converted_audio_base64 = rvc_response_json.get("converted_audio_base64")

            if not converted_audio_base64:
                logger.error(f"RVCサービスからキャラクター {current_char.character_name} のconverted_audio_base64が返却されませんでした。")
                raise RuntimeError("RVCサービスからの変換後音声が空です。")

            logger.info(f"RVC変換後音声のBase64長: {len(converted_audio_base64)}")
            converted_audio_bytes = base64.b64decode(converted_audio_base64)

            if not converted_audio_bytes:
                logger.error(f"キャラクター {current_char.character_name} のRVC音声バイトデータがデコード後に空になりました。")
                raise RuntimeError("RVC変換後音声データがデコード後に空です。")
            else:
                logger.info(f"デコードされたRVC音声バイトサイズ: {len(converted_audio_bytes)} バイト")

            return base64.b64encode(converted_audio_bytes).decode("utf-8"), emotion_label

        except requests.exceptions.RequestException as e:
            logger.error(f"キャラクター {current_char.character_name} のRVCサービス呼び出しエラー: {e}")
            if hasattr(rvc_response, 'content'):
                logger.error(f"RVC エラーレスポンス: {rvc_response.text}")
            raise
        except Exception as e:
            logger.error(f"キャラクター {current_char.character_name} のRVC変換中に予期せぬエラーが発生しました: {e}")
            raise

    async def generate_audio(
        self,
        text: str,
        character_index: int,
        audio_language_choice: str,
        sakiko_state: bool,
        synthesis_method: str = "gpt_sovits"
    ) -> tuple[str, str]:
        """指定された合成方法に基づいて音声を生成・ディスパッチする"""
        if synthesis_method == "gpt_sovits":
            return await self._generate_audio_gpt_sovits(text, character_index, audio_language_choice, sakiko_state)
        elif synthesis_method == "tts_rvc":
            return await self._generate_audio_tts_rvc(text, character_index, audio_language_choice)
        elif synthesis_method == "edge_tts_rvc":
            return await self._generate_audio_edge_tts_rvc(text, character_index, audio_language_choice)
        elif synthesis_method == "azure_tts_rvc":
            return await self._generate_audio_azure_tts_rvc(text, character_index, audio_language_choice)
        else:
            raise ValueError(f"不明な合成方式です: {synthesis_method}。'gpt_sovits', 'tts_rvc', 'edge_tts_rvc', または 'azure_tts_rvc' を指定してください。")


# FastAPI ルーティングとエンドポイント定義
audio_router = APIRouter()

class SynthesizeAudioSegmentRequest(BaseModel):
    text_segment: str
    character_index: int
    audio_language_choice: str
    sakiko_state: Optional[bool] = False
    synthesis_method: Optional[str] = "gpt_sovits"

class InitializeModelRequest(BaseModel):
    character_index: int

_current_audio_generator: SimpleAudioGenerator = None

def get_audio_generator():
    """現在のAudioGeneratorインスタンスを取得する依存関係関数"""
    global _current_audio_generator
    if _current_audio_generator is None:
        raise RuntimeError("SimpleAudioGeneratorが初期化されていません (audio_api_service.py)")
    return _current_audio_generator

def set_audio_generator_instance(generator: SimpleAudioGenerator):
    """AudioGeneratorインスタンスを設定する"""
    global _current_audio_generator
    _current_audio_generator = generator

def set_redis_client(client: Any):
    """Redisクライアントを設定する"""
    global _redis_client
    _redis_client = client

def set_emotion_model(model: Any):
    """感情分析モデルを設定する"""
    global _global_emotion_model
    _global_emotion_model = model


@audio_router.post("/initialize_gptsovits_model")
async def initialize_gptsovits_model_endpoint(
    request: InitializeModelRequest,
    audio_generator: SimpleAudioGenerator = Depends(get_audio_generator)
):
    """
    モデル初期化リクエストのエンドポイント。
    新しいアーキテクチャでは、GPU Workerがモデルを常駐化させている（Zero Cold Start）ため、
    Gateway側では成功応答のみを返却し、フロントエンドのフローを維持します。
    """
    logger.info(f"キャラクターインデックス {request.character_index} の初期化リクエストを受信しました。モデルはGPU Worker側で管理されています。")
    return {"message": f"キャラクターインデックス {request.character_index} のモデルはGPU Worker側で準備完了しています。"}


@audio_router.post("/synthesize_audio_segment")
async def synthesize_audio_segment(
    request: SynthesizeAudioSegmentRequest,
    audio_generator: SimpleAudioGenerator = Depends(get_audio_generator)
):
    """音声セグメント合成のエンドポイント"""
    try:
        audio_base64, emotion_label = await audio_generator.generate_audio(
            text=request.text_segment,
            character_index=request.character_index,
            audio_language_choice=request.audio_language_choice,
            sakiko_state=request.sakiko_state,
            synthesis_method=request.synthesis_method
        )
        return {"text_segment": request.text_segment, "audio_base64": audio_base64, "emotion": emotion_label}
    except Exception as e:
        logger.exception("音声セグメントの合成中にエラーが発生しました:")
        
        # エラー発生時は静音（無音）オーディオをフォールバックとして返す
        silent_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "reference_audio", "silent_audio", "silence.wav")
        if not os.path.exists(silent_audio_path):
            sample_rate = 22050
            duration = 1
            silent_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
            os.makedirs(os.path.dirname(silent_audio_path), exist_ok=True)
            sf.write(silent_audio_path, silent_data, sample_rate)
        
        with open(silent_audio_path, "rb") as f:
            silent_audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        return {
            "text_segment": request.text_segment, 
            "audio_base64": silent_audio_base64, 
            "error": str(e), 
            "emotion": "neutral"
        }