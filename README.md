


<div align="center">
  <a href="#jp">🇯🇵 日本語</a> | <a href="#en">🇺🇸 English</a>
</div>

<span id="jp"></span>

# AI/Live2D Character Chat - API Gateway

## プロジェクト構成について (Project Structure)

本リポジトリは、「AI/Live2D キャラクターチャットアプリケーション」における **API Gateway（バックエンドの司令塔）** を担当するプロジェクトです。

システム全体は、負荷に応じた独立したスケーリング（CPU/GPU）を実現するため、以下の4つのリポジトリに分割されています：

*   **Backend - API Gateway: 本リポジトリ**
    *   役割: フロントエンドからのHTTPリクエスト受付、LLM（Gemini/Qwen）連携、正規表現を用いた感情タグ抽出、外部TTS APIの呼び出し、およびRedisを通じたGSV Workerへのタスクルーティング。
*   [Backend - GSV Service (System Main Docs)](https://github.com/3dayspark/Live2DChat_GSV_Service)
    *   役割: システム全体のアーキテクチャ詳細、およびGPUを利用したGPT-SoVITS音声合成タスクの非同期処理。
*   [Frontend - Live2D & Vue.js](https://github.com/3dayspark/Live2DChat_Vue)
    *   役割: Live2Dの描画、チャットUI、リップシンク制御。
*   [Microservice - RVC Service](https://github.com/3dayspark/Live2DChat_RVC_Service)
    *   役割: 外部TTSから取得した音声のリアルタイム声質変換。

## 完全なドキュメントについて (Full Documentation)

プロジェクトの全体アーキテクチャ、Redisを利用した非同期処理の仕組み、マルチLLM/マルチTTS機能、およびアーキテクチャの変遷（Update Log）に関する詳細は、メインドキュメントである **[GSV Service (System Main Docs)]** のREADMEをご覧ください。

**[完全なREADMEを見る (Visit Main Documentation)](https://github.com/3dayspark/Live2DChat_GSV_Service)**

---

<span id="en"></span>

# AI/Live2D Character Chat - API Gateway

## Project Structure

This repository acts as the **API Gateway** for the "AI/Live2D Character Chat Application". It is a lightweight, CPU-bound service designed to orchestrate the backend operations.

**Roles of this Gateway:**
* Handles HTTP requests from the frontend.
* Communicates with LLMs (Gemini / ModelScope Qwen).
* Extracts emotion tags via Regex (replacing heavy local NLP models).
* Calls external TTS APIs (Gemini/Edge/Azure) and sends data to the RVC Service.
* Routes heavy GPT-SoVITS audio synthesis tasks to the GPU Worker via Redis queues.

## Full Documentation

For the complete system architecture, Redis integration details, multi-LLM/Multi-TTS features, and the recent architectural update log, please refer to the Main Documentation located in the **GSV Service** repository.

**[Visit Main Documentation](https://github.com/3dayspark/Live2DChat_GSV_Service)**







