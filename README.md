# 智慧問答關鍵字查詢 API

這是一個基於 Flask 開發的後端應用程式，旨在提供一個智慧的 AI 查詢 API。它接收使用者的自然語言問題和一個預設的關鍵字列表，並利用大型語言模型（LLM）來分析問題，找出與問題最相關的關鍵字。

此專案的設計初衷是為「國立勤益科技大學」新生提供一個快速找到問題解答的管道，例如，當新生詢問「住哪裡比較方便？」時，系統能自動回傳「5-學生宿舍」和「6-校外租屋」這兩個最相關的項目。

## 核心功能

- **雙 AI 引擎支援**：透過串接 Ollama 與 Google Gemini 服務，能夠理解使用者以繁體中文提出的問題。
- **動態關鍵字關聯**：根據問題的語意，從給定的關鍵字清單中智慧篩選出相關項目。
- **彈性的架構設計**：採用策略模式（Strategy Pattern）將 AI 服務（Ollama、Gemini）模組化，未來可以輕易擴充或替換。
- **穩健的資料處理**：具備一套清晰的資料處理管道，包含提示詞建構、AI 回應解析、格式驗證與標準化。
- **簡易的前端介面**：提供兩個基本頁面 (`index.html` 和 `voice.html`)，用於快速測試 API 功能。

## 架構設計

本專案採用了基於「策略模式」與「工廠模式」的物件導向設計，以應對未來可能需要串接多種不同 AI 模型的變化。同時，在資料處理流程上，也借鑒了函式編程（Functional Programming）風格，將各個處理步驟拆解為獨立的純函式，以兼顧程式碼的清晰度與擴充性。

### 核心設計理念

我們將「呼叫 AI 模型」這項任務抽象化，讓主程式邏輯與具體的 AI 服務（例如 Ollama 或 Google Gemini）分離。

### 主要構成部分

1.  **`BaseAIService` (抽象基礎類別)**
    *   **角色**：這是一份「合約」或「藍圖」。
    *   **作用**：它使用 `abc` 模組規定了所有 AI 服務都必須提供一個名為 `get_ai_keywords` 的方法。這確保了無論我們未來串接哪種 AI，它們的呼叫方式都是統一的。

2.  **`OllamaService`, `GeminiService` (具體的服務類別)**
    *   **角色**：這些是「合約」的具體實作者。
    *   **作用**：每一個類別都封裝了與特定 AI（如 Ollama 或 Gemini）溝通的所有細節，包括 API 的網址、金鑰驗證、請求的格式、以及如何解析其獨特的回應。

3.  **`get_ai_service()` (工廠函式)**
    *   **角色**：這是一個生產 AI 服務物件的「工廠」。
    *   **作用**：主程式不直接建立 `OllamaService` 或 `GeminiService` 物件，而是告訴這個工廠它需要哪一種服務（例如 "ollama"），工廠則回傳一個對應的物件。這讓主程式保持乾淨，未來更換 AI 模型時，可能只需要修改這個工廠函式即可。

4.  **`ai_query()` (主程式邏輯)**
    *   **角色**：這是整個流程的「總指揮」。
    *   **作用**：它的職責很單純：
        a. 接收前端請求。
        b. 向「工廠」要一個 AI 服務物件。
        c. 使用這個物件來呼叫 AI (它不關心物件到底是 Ollama 還是 Gemini)。
        d. 對 AI 回傳的結果進行驗證與標準化。
        e. 將最終結果回傳給前端。

## 環境需求

- Python 3.7+
- [Flask](https://flask.palletsprojects.com/)
- [google-generativeai](https://pypi.org/project/google-generativeai/)
- (可選) [Ollama](https://ollama.com/) 本地端服務
- (可選) 一個已下載的 Ollama 模型（程式預設使用 `gemma3:4b`）
- (可選) 一組 Google Gemini API 金鑰

## 安裝與啟動

1.  **複製專案**
    ```bash
    git clone <repository-url>
    cd qrag
    ```

2.  **建立並啟用虛擬環境** (建議)
    ```bash
    # Windows
    python -m venv env
    .\env\Scripts\activate

    # macOS / Linux
    python3 -m venv env
    source env/bin/activate
    ```

3.  **安裝相依套件**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(可選) 設定 Gemini API 金鑰**
    若要使用 Gemini 服務，請在專案根目錄下建立一個名為 `.env` 的檔案，並在其中填入您的 API 金鑰：
    ```
    GEMINI_API_KEY="AIzaSy...YOUR_API_KEY"
    ```

5.  **(可選) 設定並啟動 Ollama 服務**
    若要使用 Ollama，請確保您的 Ollama 應用程式正在背景執行。

    a. **下載模型**
        ```bash
        ollama pull gemma3:4b
        ```

    b. **設定環境變數 (重要)**
        為了讓 Flask 應用程式可以順利連接到在本機執行的 Ollama 服務，您可能需要設定以下環境變數，特別是當您使用 Docker 或虛擬機時。

        - **Windows (以系統管理員身分)**:
          ```bash
          # 設定 OLLAMA_HOST 允許多重來源連線
          setx OLLAMA_HOST "0.0.0.0"
          # 重新啟動您的終端機或電腦以使設定生效
          ```

        - **macOS / Linux**:
          ```bash
          # 設定 OLLAMA_HOST 允許多重來源連線
          export OLLAMA_HOST=0.0.0.0
          # 若要讓此設定永久生效，請將上述指令加入您的 shell 設定檔中 (例如 ~/.bashrc 或 ~/.zshrc)
          ```

6.  **啟動 Flask 應用程式**
    ```bash
    flask --debug run
    ```
    當您看到 `AI查詢服務已啟動...` 的訊息時，表示伺服器已成功運行。

7.  **開啟瀏覽器**
    在瀏覽器中開啟 `http://127.0.0.1:5000` 即可看到前端測試頁面。

## API 端點說明

### `POST /api/ai-query`

此為本專案的核心 API，用於處理 AI 查詢請求。

- **請求格式** (`application/json`)

    ```json
    {
      "question": "我想問學校宿舍的申請辦法，還有費用怎麼算？",
      "keywords": [
        "1-註冊須知(含大一英文分班資訊)",
        "2-新生入學登錄",
        "3-新生體檢",
        "4-學號查詢",
        "5-學生宿舍(新生入住須知)",
        "6-校外租屋",
        "7-機車停車位申請",
        "8-新生定向輔導",
        "9-學雜費減免",
        "10-就學貸款",
        "11-獎助學金資訊",
        "12-弱勢助學計畫",
        "13-校內工讀",
        "14-資訊系統學生篇",
        "15-學生綜合資料",
        "16-第e學雜費入口網",
        "17-交通資訊",
        "18-校園地圖",
        "19-行事曆",
        "20-學生手冊",
        "21-3+1就學役男"
      ]
    }
    ```

- **成功回應** (`200 OK`)

    ```json
    {
      "relevant_keywords": [
        "5-學生宿舍(新生入住須知)"
      ]
    }
    ```

- **錯誤回應**

    若請求格式錯誤、AI 服務無法連線或發生其他伺服器內部錯誤，將回傳對應的錯誤訊息。
    ```json
    {
      "error": "無法連接至 AI 服務: <詳細錯誤訊息>"
    }
    ```

## 如何擴充新的 AI 模型？

若未來要新增一個例如 `NewAIService` 的模型，只需三步驟：

1.  建立一個新的類別 `class NewAIService(BaseAIService):`。
2.  確保在這個新類別中，完整實作 `get_ai_keywords` 方法，包含呼叫新 AI API 的所有邏輯。
3.  在 `get_ai_service` 工廠函式中，增加一個 `elif service_name == "newai": return NewAIService()` 的選項。

完成後，整個系統就能在不更動主程式邏輯的情況下，無縫支援新的 AI 模型。

此外，您也可以透過在 API 請求中加入 `"service": "gemini"` 參數來指定要使用的 AI 服務。目前的網頁前端預設使用 `ollama`，但您可以透過 Postman 或其他工具來測試此功能。

## 疑難排解

- **無法連接至 AI 服務 (Connection Error)**
  - **Ollama**:
    1. 請確認 Ollama 應用程式正在本機執行。
    2. 確認 `OLLAMA_HOST` 環境變數是否已正確設定為 `0.0.0.0`。
    3. 檢查您的防火牆或防毒軟體是否阻擋了應用程式對 `http://localhost:11434` 的連線。
  - **Gemini**:
    1. 請確認專案根目錄下已有 `.env` 檔案，且 `GEMINI_API_KEY` 已正確填寫。
    2. 檢查您的網路連線是否正常，以及是否能存取 Google API 服務。
