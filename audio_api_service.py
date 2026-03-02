import base64
import re
import os
import json
import time
import struct
import logging
import asyncio
import uuid
from typing import List, Dict, Optional, Any
from xml.etree import ElementTree
from dotenv import load_dotenv

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import soundfile as sf
import numpy as np
import requests
import edge_tts

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api_character_loader import CharacterAttributes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_redis_client: Any = None

GEMINI_TTS_URL = "https://asynchronousblocking.asia/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"

load_dotenv()
RVC_SERVICE_URL = os.environ.get("RVC_SERVICE_URL", "http://127.0.0.1:8001/rvc_convert")

AZURE_TTS_REGION = "japanwest"
AZURE_TTS_ENDPOINT = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
AZURE_TTS_TOKEN_URL = f"https://{AZURE_TTS_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
AZURE_TTS_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
AZURE_TTS_OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"
AZURE_TTS_USER_AGENT = "Chat_backend"

def parse_audio_mime_type(mime_type: str) -> dict[str, Optional[int]]:
    bits_per_sample = 16
    rate = 24000
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
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
    def __init__(self, character_list: List[CharacterAttributes]):
        self.character_list = character_list
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.replacements_jap = {
            '豊川祥子': 'とがわさきこ', '祥子': 'さきこ', '三角初華': 'みすみういか', '初華': 'ういか',
            '若葉睦': 'わかばむつみ', '睦': 'むつみ', '八幡海鈴': 'やはたうみり', '海鈴': 'うみり',
            '海铃': 'うみり', '祐天寺': 'ゆうてんじ', '若麦': 'にゃむ', '喵梦': 'にゃむ',
            '高松燈': 'たかまつともり', '燈': 'ともり', '灯': 'ともり', '椎名立希': 'しいなたき',
            '莉莎': 'リサ', '愛音': 'アノン', '素世': 'そよ', '爽世': 'そよ', '千早愛音': 'ちはやアノン',
            '爱音': 'アノン', '要楽奈': 'かなめらーな', '楽奈': 'らーな', '春日影': 'はるひかげ',
            'Doloris': 'ドロリス', 'Mortis': 'モーティス', 'Timoris': 'ティモリス', 'Amoris': 'アモーリス',
            'Oblivionis': 'オブリビオニス', 'live': 'ライブ', 'RiNG': 'リング', '珠手知由': 'たまでちゆ',
            'CHUCHU': 'チュチュ', 'CHU²': 'チュチュ', 'CHU2': 'チュチュ', '友希那': 'ゆきな',
            '纱夜': 'サヨ', '牛肉干': 'ジャーキー', 'Roselia': 'ロゼリア', '垃圾桶': 'ゴミ箱',
            '髪型': 'かみがた', 'RAISE A SUILEN': 'レイズアスイレン', 'Senior Yukina': 'ゆきな先輩',
            'MyGO!!!!!': 'まいご'
        }
        
        self.replacements_chi = {
            'CRYCHIC': 'C团', 'live': "演出", 'RiNG': "ring", 'Doloris': '初华',
            'Mortis': '睦', 'Timoris': '海铃', 'Amoris': '喵梦', 'Oblivionis': '我',
            'MyGO': 'mygo', 'ちゃん': '', 'CHU²': '楚楚', 'CHU2': '楚楚'
        }
        
        self.replacements_yue = {
            '丰川祥子': 'fung1 cyun1 coeng4 zi2', '祥子': 'coeng4 zi2', '睦': 'muk6',
            '爱音': 'oi3 jam1', '千早爱音': 'cin1 zou2 oi3 jam1', '立希': 'lap6 hei1',
            '椎名立希': 'ceoi1 naa4 lap6 hei1', '爽世': 'song2 sai3', '要乐奈': 'jiu3 lok6 naa4',
            '乐奈': 'lok6 naa4', '春日影': 'har1 jat6 jing2', 'CRYCHIC': 'klai4 sik1',
            'Ave Mujica': 'aai1 wai1 muk6 zi1 gaa3', 'Doloris': 'do1 lo4 lei6 si1',
            'Mortis': 'muk6 ti4 si1', 'Timoris': 'ti4 mo1 lei6 si1', 'Amoris': 'aa3 mo1 lei6 si1',
            'Oblivionis': 'o1 bi1 lip6 bi1 o1 nis1', 'live': 'laai1 fuk6', 'MyGO': 'mai5 go1', 'RiNG': 'ring1',
        }

        self.GEMINI_API_KEY = None
        self.generated_audio_folder = os.path.join(base_dir, "reference_audio", "generated_audios_temp")
        os.makedirs(self.generated_audio_folder, exist_ok=True)

        self.EDGE_TTS_DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
        self.EDGE_TTS_DEFAULT_RATE = "+0%"
        self.EDGE_TTS_DEFAULT_VOLUME = "+0%"

        self.AZURE_TTS_SUBSCRIPTION_KEY = None
        self.azure_tts_access_token = None
        self.azure_tts_token_expiry_time = 0

        self.edge_tts_lock = asyncio.Lock()

    async def load_gemini_api_key(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        key_file_path = os.path.join(script_dir, "API Key.txt")
        try:
            with open(key_file_path, "r", encoding="utf-8") as f:
                self.GEMINI_API_KEY = f.read().strip()
            logger.info("Gemini API Keyの読み込みに成功しました。")
        except FileNotFoundError:
            logger.error(f"エラー: API Keyファイルが {key_file_path} に見つかりません。")
            self.GEMINI_API_KEY = None
        except Exception as e:
            logger.error(f"Gemini API Keyの読み込み中にエラーが発生しました: {e}")
            self.GEMINI_API_KEY = None

    async def load_azure_tts_subscription_key(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        azure_key_file_path = os.path.join(script_dir,  "API Key_Azure.txt")
        try:
            with open(azure_key_file_path, "r", encoding="utf-8") as f:
                azure_key = f.read().strip()
                if azure_key:
                    self.AZURE_TTS_SUBSCRIPTION_KEY = azure_key
                    logger.info("Azure TTS Subscription Keyの読み込みに成功しました。")
                else:
                    logger.warning(f"'{azure_key_file_path}' は空です。")
        except FileNotFoundError:
            logger.error(f"エラー: Azure TTS Keyファイルが {azure_key_file_path} に見つかりません。")
            self.AZURE_TTS_SUBSCRIPTION_KEY = None
        except Exception as e:
            logger.error(f"Azure TTS Subscription Keyの読み込み中にエラーが発生しました: {e}")
            self.AZURE_TTS_SUBSCRIPTION_KEY = None

    async def _get_azure_tts_access_token(self):
        if not self.AZURE_TTS_SUBSCRIPTION_KEY:
            raise RuntimeError("Azure TTS Subscription Keyがロードされていません。")
        if self.azure_tts_access_token and time.time() < self.azure_tts_token_expiry_time:
            return self.azure_tts_access_token

        try:
            headers = {'Ocp-Apim-Subscription-Key': self.AZURE_TTS_SUBSCRIPTION_KEY}
            response = requests.post(AZURE_TTS_TOKEN_URL, headers=headers, timeout=10)
            response.raise_for_status()
            self.azure_tts_access_token = str(response.text)
            self.azure_tts_token_expiry_time = time.time() + 9 * 60 
            return self.azure_tts_access_token
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Azure TTS Access Tokenの取得に失敗しました: {e}")

    def _preprocess_text(self, text: str, character_index: int, audio_language_choice: str) -> str:

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
        else:
            for key, value in self.replacements_chi.items():
                processed_text = re.sub(re.escape(key), value, processed_text, flags=re.IGNORECASE)


        pattern = r'^[^A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]+\'\''
        processed_text = re.sub(pattern, '', processed_text).replace(' ', '')

        valid_char_pattern = re.compile(r'[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]')

        if not processed_text or not valid_char_pattern.search(processed_text):
            return '今年'
            
        return processed_text

    def _apply_rvc_conversion(self, audio_wav_bytes: bytes, rvc_model_dir_id: str, rvc_index_dir_id: str, **kwargs) -> str:
        payload = {
            "audio_data_base64": base64.b64encode(audio_wav_bytes).decode('utf-8'),
            "rvc_model_relative_dir": rvc_model_dir_id, 
            "rvc_index_relative_dir": rvc_index_dir_id,
            "pitch_shift": 0, "f0_method": "rmvpe", "index_rate": 0.75, 
            "filter_radius": 3, "resample_sr": 0, "rms_mix_rate": 0.25, "protect": 0.33
        }
        payload.update(kwargs)

        rvc_headers = {"Content-Type": "application/json"}
        try:
            rvc_response = requests.post(RVC_SERVICE_URL, headers=rvc_headers, json=payload, timeout=60)
            rvc_response.raise_for_status()
            converted_audio_base64 = rvc_response.json().get("converted_audio_base64")
            if not converted_audio_base64:
                raise RuntimeError("RVCサービスからの変換後音声が空です。")
            converted_audio_bytes = base64.b64decode(converted_audio_base64)
            return base64.b64encode(converted_audio_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"RVC変換中にエラーが発生しました: {e}")
            raise

    async def _generate_audio_gpt_sovits(
        self, text: str, character_index: int, audio_language_choice: str, sakiko_state: bool
    ) -> tuple[str, str]:
        if _redis_client is None:
            raise RuntimeError("Redisクライアントが初期化されていません。")

        current_char = self.character_list[character_index]
        processed_text = self._preprocess_text(text, character_index, audio_language_choice)

        task_id = str(uuid.uuid4())
        task_payload = {
            "task_id": task_id,
            "text": processed_text,
            "character_folder": current_char.character_folder_name,
            "audio_language_choice": audio_language_choice,
            "sakiko_state": sakiko_state,
        }

        queue_name = "queue:audio:global"
        try:
            _redis_client.rpush(queue_name, json.dumps(task_payload))
            logger.info(f"タスク {task_id} をグローバルキュー {queue_name} にプッシュしました。")
        except Exception as e:
            raise RuntimeError(f"タスクのディスパッチに失敗しました: {e}")

        result_key = f"result:{task_id}"
        audio_base64 = None
        
        for _ in range(600):
            result_data = _redis_client.get(result_key)
            if result_data:
                res = json.loads(result_data)
                if "error" in res:
                    raise RuntimeError(f"GPU Workerでの合成に失敗しました: {res['error']}")
                audio_base64 = res.get("audio_base64")
                _redis_client.delete(result_key)
                break
            await asyncio.sleep(0.1)

        if not audio_base64:
            raise TimeoutError("GPU Workerからの応答がタイムアウトしました。")

        return audio_base64, "neutral"

    async def _generate_audio_tts_rvc(self, text: str, character_index: int, audio_language_choice: str) -> tuple[str, str]:
        current_char = self.character_list[character_index]
        rvc_model_dir_id = current_char.rvc_model_dir_id
        rvc_index_dir_id = current_char.rvc_index_dir_id
        if not rvc_model_dir_id or not rvc_index_dir_id:
            raise ValueError("RVCモデル/インデックスディレクトリIDが設定されていません。")

        if not self.GEMINI_API_KEY:
            raise RuntimeError("Gemini API Keyが読み込まれていません。")

        processed_text = self._preprocess_text(text, character_index, audio_language_choice)

        tts_payload = {
            "contents": [{"parts": [{"text": processed_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}}}
            },
            "model": "gemini-2.5-flash-preview-tts",
        }
        tts_headers = {"Content-Type": "application/json", "X-goog-api-key": self.GEMINI_API_KEY}

        try:
            response = requests.post(GEMINI_TTS_URL, headers=tts_headers, json=tts_payload, timeout=30)
            response.raise_for_status()
            response_json = response.json()

            if "error" in response_json:
                raise RuntimeError(f"Gemini TTS API エラー: {response_json['error']}")

            base64_audio_data = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("inlineData", {}).get("data")
            if not base64_audio_data:
                raise RuntimeError("Gemini TTSからの音声データが空です。")

            tts_wav_data = convert_to_wav(base64.b64decode(base64_audio_data), "audio/L16;rate=24000")
        except Exception as e:
            raise RuntimeError(f"Gemini TTS予期せぬエラー: {e}")

        converted_b64 = self._apply_rvc_conversion(tts_wav_data, rvc_model_dir_id, rvc_index_dir_id)
        return converted_b64, "neutral"

    async def _generate_audio_edge_tts_rvc(self, text: str, character_index: int, audio_language_choice: str) -> tuple[str, str]:
        current_char = self.character_list[character_index]
        rvc_model_dir_id = current_char.rvc_model_dir_id
        rvc_index_dir_id = current_char.rvc_index_dir_id
        if not rvc_model_dir_id or not rvc_index_dir_id:
            raise ValueError("RVCモデル/インデックスディレクトリIDが設定されていません。")

        edge_tts_voice = self.EDGE_TTS_DEFAULT_VOICE
        base_dir = os.path.dirname(os.path.abspath(__file__))
        character_folder_path = os.path.join(base_dir, "reference_audio", current_char.character_folder_name)
        azure_voice_config_path = os.path.join(character_folder_path, "azure_voice.txt")
        custom_voices =[]

        if os.path.exists(azure_voice_config_path):
            with open(azure_voice_config_path, "r", encoding="utf-8") as f:
                custom_voices =[line.strip() for line in f.readlines() if line.strip()]

        if custom_voices:
            if audio_language_choice == "日英混合" and len(custom_voices) >= 2:
                edge_tts_voice = custom_voices[1]
            elif audio_language_choice == "粤英混合" and len(custom_voices) >= 3:
                edge_tts_voice = custom_voices[2]
            elif len(custom_voices) >= 1:
                edge_tts_voice = custom_voices[0]

        if audio_language_choice == "日英混合" and (not custom_voices or len(custom_voices) < 2):
            edge_tts_voice = "ja-JP-NanamiNeural"  # Edge不支持AoiNeural
        elif audio_language_choice == "粤英混合" and (not custom_voices or len(custom_voices) < 3):
            edge_tts_voice = "zh-HK-HiuMaanNeural" # Edge大概率不支持HiuGaaiNeural

        processed_text = self._preprocess_text(text, character_index, audio_language_choice)


        logger.info(f"[Edge TTS 准备合成] 音色: '{edge_tts_voice}' | 文本: '{processed_text}'")

        try:
            communicate = edge_tts.Communicate(
                processed_text,
                edge_tts_voice,
                rate=self.EDGE_TTS_DEFAULT_RATE,
                volume=self.EDGE_TTS_DEFAULT_VOLUME,
            )
            audio_data_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data_chunks.append(chunk["data"])
            
            if not audio_data_chunks:
                raise edge_tts.exceptions.NoAudioReceived("未接收到任何音频数据")
                
            edge_tts_wav_data = b"".join(audio_data_chunks)
            
        except edge_tts.exceptions.NoAudioReceived as e:
            logger.error(f"[Edge TTS 被拒绝] 音色 '{edge_tts_voice}' 可能不受Edge免费版支持，或者原文本无效！")
            

            if audio_language_choice == "日英混合":
                backup_voice = "ja-JP-NanamiNeural"
                backup_text = "音声合成に失敗しました"
            elif audio_language_choice == "粤英混合":
                backup_voice = "zh-HK-HiuMaanNeural"
                backup_text = "發音合成發生錯誤"
            else:
                backup_voice = "zh-CN-XiaoxiaoNeural"
                backup_text = "发音合成发生错误"
                
            logger.info(f"--> [兜底阶段 1] 尝试使用免费保底音色 '{backup_voice}' 朗读原文本...")
            try:

                communicate = edge_tts.Communicate(
                    processed_text,
                    backup_voice,
                    rate=self.EDGE_TTS_DEFAULT_RATE,
                    volume=self.EDGE_TTS_DEFAULT_VOLUME,
                )
                audio_data_chunks =[]
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data_chunks.append(chunk["data"])
                        
                if not audio_data_chunks:
                    raise edge_tts.exceptions.NoAudioReceived("保底音色朗读原文本依然未收到音频")
                    
                edge_tts_wav_data = b"".join(audio_data_chunks)
                logger.info("-->[兜底阶段 1] 成功！原文本已被保底音色合成。")
                
            except edge_tts.exceptions.NoAudioReceived:

                logger.warning(f"--> [兜底阶段 2] 原文本彻底无法发音，正在朗读固定报错文案...")
                try:

                    communicate = edge_tts.Communicate(backup_text, backup_voice)
                    audio_data_chunks = []
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data_chunks.append(chunk["data"])
                            
                    if not audio_data_chunks:
                        raise RuntimeError("终极兜底失败，未生成任何音频数据")
                        
                    edge_tts_wav_data = b"".join(audio_data_chunks)
                    logger.info("--> [兜底阶段 2] 成功！已生成报错提示音。")
                    
                except Exception as backup_e:
                    raise RuntimeError(f"Edge TTS 终极兜底重试也失败了: {backup_e}")
            except Exception as e:
                raise RuntimeError(f"Edge TTS 兜底阶段 1 发生异常: {e}")

        except Exception as e:
            raise RuntimeError(f"Edge TTS 发生异常: {e}")

        converted_b64 = self._apply_rvc_conversion(edge_tts_wav_data, rvc_model_dir_id, rvc_index_dir_id)
        return converted_b64, "neutral"

    async def _generate_audio_azure_tts_rvc(self, text: str, character_index: int, audio_language_choice: str) -> tuple[str, str]:
        current_char = self.character_list[character_index]
        rvc_model_dir_id = current_char.rvc_model_dir_id
        rvc_index_dir_id = current_char.rvc_index_dir_id
        if not rvc_model_dir_id or not rvc_index_dir_id:
            raise ValueError("RVCモデル/インデックスディレクトリIDが設定されていません。")

        if not self.AZURE_TTS_SUBSCRIPTION_KEY:
            raise RuntimeError("Azure TTS Subscription Keyが読み込まれていません。")

        processed_text = self._preprocess_text(text, character_index, audio_language_choice)

        try:
            access_token = await self._get_azure_tts_access_token()
            azure_tts_voice_name = AZURE_TTS_DEFAULT_VOICE
            base_dir = os.path.dirname(os.path.abspath(__file__))
            character_folder_path = os.path.join(base_dir, "reference_audio", current_char.character_folder_name)
            azure_voice_config_path = os.path.join(character_folder_path, "azure_voice.txt")
            custom_voices =[]

            if os.path.exists(azure_voice_config_path):
                with open(azure_voice_config_path, "r", encoding="utf-8") as f:
                    custom_voices =[line.strip() for line in f.readlines() if line.strip()]

            if custom_voices:
                if audio_language_choice == "日英混合" and len(custom_voices) >= 2:
                    azure_tts_voice_name = custom_voices[1]
                elif audio_language_choice == "粤英混合" and len(custom_voices) >= 3:
                    azure_tts_voice_name = custom_voices[2]
                elif len(custom_voices) >= 1:
                    azure_tts_voice_name = custom_voices[0]

            xml_body = ElementTree.Element('speak', version='1.0')
            voice = ElementTree.SubElement(xml_body, 'voice')

            if audio_language_choice == "日英混合":
                if not custom_voices or len(custom_voices) < 2: azure_tts_voice_name = "ja-JP-AoiNeural"
                xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', 'ja-JP')
                voice.set('{http://www.w3.org/XML/1998/namespace}lang', 'ja-JP')
            elif audio_language_choice == "粤英混合":
                if not custom_voices or len(custom_voices) < 3: azure_tts_voice_name = "zh-HK-HiuGaaiNeural"
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
        except Exception as e:
            raise RuntimeError(f"Azure TTS予期せぬエラー: {e}")

        custom_kwargs = {
            "f0_up_key": 0, "protect": 0.5, "rms_mix_rate": 1, "tuner_steps": 200
        }
        converted_b64 = self._apply_rvc_conversion(azure_tts_audio_data, rvc_model_dir_id, rvc_index_dir_id, **custom_kwargs)
        return converted_b64, "neutral"

    async def generate_audio(
        self, text: str, character_index: int, audio_language_choice: str, sakiko_state: bool, synthesis_method: str = "gpt_sovits"
    ) -> tuple[str, str]:
        if synthesis_method == "gpt_sovits":
            return await self._generate_audio_gpt_sovits(text, character_index, audio_language_choice, sakiko_state)
        elif synthesis_method == "tts_rvc":
            return await self._generate_audio_tts_rvc(text, character_index, audio_language_choice)
        elif synthesis_method == "edge_tts_rvc":
            return await self._generate_audio_edge_tts_rvc(text, character_index, audio_language_choice)
        elif synthesis_method == "azure_tts_rvc":
            return await self._generate_audio_azure_tts_rvc(text, character_index, audio_language_choice)
        else:
            raise ValueError(f"不明な合成方式です: {synthesis_method}")

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
    global _current_audio_generator
    if _current_audio_generator is None:
        raise RuntimeError("SimpleAudioGeneratorが初期化されていません")
    return _current_audio_generator

def set_audio_generator_instance(generator: SimpleAudioGenerator):
    global _current_audio_generator
    _current_audio_generator = generator

def set_redis_client(client: Any):
    global _redis_client
    _redis_client = client

@audio_router.post("/initialize_gptsovits_model")
async def initialize_gptsovits_model_endpoint(
    request: InitializeModelRequest,
    audio_generator: SimpleAudioGenerator = Depends(get_audio_generator)
):
    return {"message": f"キャラクターインデックス {request.character_index} のモデルはGPU Worker側で準備完了しています。"}

@audio_router.post("/synthesize_audio_segment")
async def synthesize_audio_segment(
    request: SynthesizeAudioSegmentRequest,
    audio_generator: SimpleAudioGenerator = Depends(get_audio_generator)
):
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