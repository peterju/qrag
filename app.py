# 檔名: app.py
# 功能: 提供一個AI查詢API，採用函式編程(FP)與物件導向(OOP)混合風格設計。
# 核心理念:
# 1. 主流程採用FP風格，將資料處理過程視為一個清晰的「轉換管道」。
# 2. AI服務部分採用OOP的「策略模式」，將Ollama、Gemini等不同服務封裝成可互換的物件，以利未來擴充。

# ------------------- 引入必要的函式庫 -------------------
import json  # 用於處理JSON格式的資料(解析與生成)
import os  # 用於處理環境變數和檔案路徑
import re  # 用於正規表示式處理，提取關鍵字中的數字編號
from abc import ABC, abstractmethod  # 用於建立抽象基礎類別 (服務合約)

import requests  # 用於向其他服務(如Ollama)發送HTTP請求
from dotenv import load_dotenv  # 用於載入環境變數，從 .env 檔案讀取敏感資訊

# Flask: 用於建立輕量級的網頁伺服器與API端點。
from flask import Flask, jsonify, render_template, request

# Flask-CORS: 解決瀏覽器"同源策略"限制，允許前端網頁(不同來源)呼叫此API。
# 在本地開發時，前端是 file:// 協議，後端是 http://，來源不同，必須使用CORS。
from flask_cors import CORS

# 嘗試引入 Gemini 函式庫，如果未安裝則設為 None，避免程式啟動失敗
try:
    import google.generativeai as genai
    from google.generativeai.client import configure
    from google.generativeai.generative_models import GenerativeModel
except ImportError:
    genai = configure = GenerativeModel = None

# 在應用程式啟動時載入 .env 檔案中的環境變數
load_dotenv()

# --- 組態設定 ---
# 決定預設使用的 AI 服務 ('ollama' 或 'gemini')
AI_SERVICE = "ollama"

# 設定不同服務的預設模型
# ollama 可使用的模型：'kenneth85/llama-3-taiwan:latest', 'gemma3:4b', 'gemma3n:e2b', 'mistral', 'granite3.3', 'qwen3:4b'
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
# gemini 可使用的模型：'gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-flash'
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
# --- 組態設定結束 ---


# ------------------- 初始化 Flask 應用 -------------------
app = Flask(__name__)  # 建立一個Flask應用實例
# 允許所有來源的跨域請求，若在正式上線的生產環境中，可以設定更嚴格的來源限制。
CORS(app)

# ==============================================================================
# --- 1. AI 服務層 (物件導向的策略模式) ---
# 這一區塊的目標是將「如何呼叫特定AI」的邏輯封裝起來，使其可以被輕易替換。
# ==============================================================================


# --- AI 服務的「合約」(抽象基礎類別) ---
class BaseAIService(ABC):
    """
    定義所有 AI 服務都必須遵守的「合約」(介面)。
    它規定了任何 AI 服務都必須提供一個名為 get_ai_keywords 的方法。
    """

    @abstractmethod
    def get_ai_keywords(self, user_question: str, original_keywords: list) -> list:
        """
        根據使用者問題和關鍵字清單，呼叫對應的 AI 模型，
        並回傳一個從 AI 回應中解析出的、未經驗證的原始關鍵字列表。
        """
        pass


# --- Ollama 服務的具體實作 ---
class OllamaService(BaseAIService):
    """使用本地 Ollama 模型來實現 AI 服務。"""

    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL):
        """初始化 Ollama 服務。"""
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        print(f"Ollama 服務已初始化，使用模型: {self.model_name}")

    def get_ai_keywords(self, user_question: str, original_keywords: list) -> list:
        # 步驟 1: 建構提示詞
        prompt = build_prompt(user_question, original_keywords)
        print(f"\n--- 正在向 Ollama ({self.model_name}) 發送請求 ---")

        # 步驟 2: 準備 API 的請求內容
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,  # 需要一次性接收完整回應
            "options": {"temperature": 0.2},  # 降低溫度，讓回答更具確定性
        }

        # 步驟 3: 呼叫 API 並處理回應
        response = requests.post(self.api_url, json=payload, timeout=90)
        response.raise_for_status()  # 若請求失敗 (如 404, 500)，會在此拋出例外
        ai_response_text = response.json().get("response", "")

        # 步驟 4: 解析回應
        return parse_ai_response(ai_response_text)


# --- Gemini 服務的具體實作 ---
class GeminiService(BaseAIService):
    """使用 Google Gemini 模型來實現 AI 服務。"""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL):
        """
        初始化 Gemini 服務，讀取並設定 API Key。
        """
        if not genai or not configure or not GenerativeModel:
            raise ImportError(
                "Gemini 服務需要 'google-generativeai' 函式庫。請先安裝: pip install google-generativeai"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("錯誤：找不到 'GEMINI_API_KEY' 環境變數。")

        # 注意: configure 是全域設定
        configure(api_key=api_key)
        self.model = GenerativeModel(
            model_name,
            # 要求 Gemini 直接回傳 JSON 格式，並設定溫度
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        )
        print(f"Gemini 服務已初始化，使用模型: {model_name}")

    def get_ai_keywords(self, user_question: str, original_keywords: list) -> list:
        # 步驟 1: 建構提示詞
        prompt = build_prompt(user_question, original_keywords)
        print(f"\n--- 正在向 Gemini ({self.model.model_name}) 發送請求 ---")

        # 步驟 2: 呼叫 API
        response = self.model.generate_content(prompt)

        # 步驟 3: 解析 JSON 回應
        # 因為已要求回傳 JSON，直接用 json.loads 解析
        # 由上層的 ai_query 統一處理可能的解析錯誤 (JSONDecodeError)
        return json.loads(response.text)


# --- AI 服務的「工廠」函式 ---
def get_ai_service(service_name: str = "ollama") -> BaseAIService:
    """
    根據指定的名稱，回傳一個對應的 AI 服務物件。
    這是典型的「工廠模式」，讓主程式不需要知道物件建立的細節。
    """
    if service_name == "gemini":
        return GeminiService()
    elif service_name == "ollama":
        return OllamaService()
    else:
        raise ValueError(f"未知的 AI 服務名稱: {service_name}")


# ==============================================================================
# --- 2. 資料處理管道 (函式編程風格的純函式) ---
# 這一區塊的目標是將每一步資料轉換都定義成一個獨立、無副作用的函式。
# ==============================================================================

PROMPT_TEMPLATE = """
你是一個非常聰明且嚴謹的大學新生資訊助理，專門服務「國立勤益科技大學」（以下簡稱「勤益」或「本校」）的新生。你的所有回答都必須使用繁體中文。
你的任務是根據使用者問題，從給定的關鍵字清單中，找出所有相關的項目。

在開始分析前，請先閱讀並記住以下的【重要背景知識與規則】，這會幫助你做出更準確的判斷。

---
【重要背景知識與規則】：
- 任何關於「怎麼去學校」、「如何到學校」、「交通」、「路線」、「地圖」的問題，都應優先考慮關聯到「17-交通資訊」和「18-校園地圖」。
- 任何關於「住宿」、「住哪裡」、「房間」的問題，都應優先考慮關聯到「5-學生宿舍(新生入住須知)」和「6-校外租屋」。
- 任何關於「錢」、「學費」、「補助」、「打工」的問題，都可能與「9-學雜費減免」、「10-就學貸款」、「11-獎助學金資訊」、「12-弱勢助學計畫」、「13-校內工讀」相關。
---

現在，請嚴格遵循以下思考步驟來完成任務：

步驟 1：【理解問題】
首先，在內心分析使用者問題的核心意圖是什麼。例如，問題「我們家的兒子今年會就讀貴校，請問報到時要注意什麼?」的核心意圖是詢問「新生報到」的所有相關流程與準備事項。

步驟 2：【逐一匹配】
接著，運用你在【重要背景知識與規則】中學到的規則，逐一檢視下方的「關鍵字清單」。對於每一個關鍵字，思考它是否與你在步驟1中分析出的核心意圖相關。

步驟 3：【彙總結果並自我審查】
將所有在步驟2中判定為「相關」的關鍵字完整地收集起來。在輸出最終結果前，再次檢查這個列表，確保沒有遺漏任何明顯相關的項目。

步驟 4：【輸出JSON】
最後，將你彙總並審查過的結果，以一個「JSON陣列」的格式輸出。你的輸出必須從 `[` 開始，以 `]` 結束。陣列中的每個關鍵字都必須是使用雙引號 `"` 包圍的字串。
例如: ["9-學雜費減免", "10-就學貸款"]
絕對不要輸出任何額外的解釋或文字，例如，不要輸出像 '好的，這是您要的JSON：[...]` 這樣的格式。如果經過分析後，確實沒有任何項目相關，則回傳一個空的陣列 `[]`。
---
【關鍵字清單】:
{keywords_str}
---
【使用者問題】:
{user_question}
---

【你的 JSON 回應】:
"""


def build_prompt(user_question: str, keywords: list) -> str:
    """(純函式) 根據使用者問題和關鍵字列表，建構一個標準化的提示詞。"""

    # 將關鍵字列表轉換為格式化的字串，並插入到提示詞模板中。
    keywords_str = "\n".join(f"- {kw}" for kw in keywords)
    return PROMPT_TEMPLATE.format(
        keywords_str=keywords_str, user_question=user_question
    )


def parse_ai_response(response_text: str) -> list:
    """(純函式) 從 AI 的原始回應中，提取並解析出 JSON 結構。"""
    print(f"AI 原始回應: {response_text}")

    # re.search 會尋找第一個匹配的模式。
    # 新的正則表達式 r'(\{[\s\S]*?\}|\[[\s\S]*?\])' 的詳細解析:
    #  -  \{[\s\S]*?\} 匹配 {...} 形式的JSON物件
    #  -  | 是 "或" 的意思
    #  -  \[[\s\S]*?\] 匹配 [...] 形式的JSON陣列
    # 這使得正則表達式可以同時捕捉這兩種常見的JSON根結構
    match = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", response_text)
    if not match:
        print("警告：未能在回應中找到 JSON 結構，將使用空陣列。")
        return []

    json_str = match.group(1)
    print(f"成功用正則表達式提取出 JSON 結構: {json_str}")

    # 預處理JSON字串，將非標準的單引號、中文引號與反斜線替換為標準的雙引號
    cleaned_json_str = (
        json_str.replace("「", '"').replace("」", '"').replace("'", '"').replace('\\"', '"')
    )
    if json_str != cleaned_json_str:
        print(f"已將非標準引號轉換為標準雙引號: {cleaned_json_str}")

    return json.loads(cleaned_json_str)


def validate_and_normalize_keywords(
    llm_answer_keywords: list, original_keywords: list
) -> list:
    """(純函式) 根據原始定義，驗證並標準化 AI 回傳的關鍵字。"""
    # 步驟1： 建立驗證用的資料結構
    # 建立儲存「完整關鍵字」的Set：{"1-註冊須知(含大一英文分班資訊)", ..., "21-3+1就學役男"}
    valid_set = set(original_keywords)  # 將原始關鍵字列表轉換為集合，便於快速查找
    # 建立二個字典來對應「編號」和「完整關鍵字」：
    # 1. id_map: 用於從編號還原到完整關鍵字 {"17": "17-交通資訊", ...}
    # 2. text_map: 用於從關鍵字純文字還原到完整關鍵字 {"交通資訊": "17-交通資訊", ...}
    # 這樣可以處理不同形式的輸入，如「17-交通資訊」、「交通資訊」或「17」等。
    id_map, text_map = {}, {}
    for kw in original_keywords:
        # 使用正則表達式 r'^(\d+)[-－]' 來匹配並提取字串開頭的數字編號和後面的分隔符
        # ^ 表示字串開頭
        # (\d+) 捕獲一個或多個數字
        # [-－] 匹配半形或全形的橫線
        match = re.match(r"^(\d+)[-－](.*)", kw)
        if match:
            id_map[match.group(1)] = kw  # 將編號和完整關鍵字存入 id_map
            text_map[match.group(2)] = kw  # 將純文字關鍵字和完整關鍵字存入 text_map

    # 步驟2： 驗證和標準化
    validated = []  # 建立一個空列表，用於存放最終驗證通過的、完整的關鍵字
    # 遍歷從AI回應中初步提取出的每一個項目 (item)
    for item in llm_answer_keywords:
        found_keyword = None
        # 情況一：AI回傳的是完整的關鍵字，且存在於合法清單中
        if isinstance(item, str) and item in valid_set:
            found_keyword = item
        # 情況二：AI回傳的是純文字關鍵字，且存在於 text_map 中
        elif isinstance(item, str) and item in text_map:
            found_keyword = text_map[item]
        # 情況三：AI回傳的是純編號 (字串或數字)，且存在於 id_map 中
        elif isinstance(item, (str, int)) and str(item) in id_map:
            found_keyword = id_map[str(item)]
        # 情況四：AI回傳的是字典，可能包含純文字關鍵字或編號
        elif isinstance(item, dict):
            possible_keys = ["key", "keyword", "項目", "關鍵字", "value"]
            for p_key in possible_keys:
                if p_key in item:
                    value_str = str(item[p_key])  # 確保轉成字串
                    # 再次進行三種核心驗證
                    if value_str in valid_set:
                        found_keyword = value_str
                        break
                    elif value_str in text_map:
                        found_keyword = text_map[value_str]
                        break
                    elif value_str in id_map:
                        found_keyword = id_map[value_str]
                        break
        if found_keyword:
            validated.append(found_keyword)
    return validated


def deduplicate_preserve_order(keywords: list) -> list:
    """(純函式) 移除列表中的重複項，同時保持原始順序。"""
    # 使用 dict.fromkeys() 而非 set() 的原因：
    # - dict.fromkeys() 保持元素的原始順序 (Python 3.7+)
    # - set() 會打亂順序，影響AI智慧排序的效果
    # - 效能相同 O(n)，但 dict.fromkeys() 更適合需要保持順序的場景
    return list(dict.fromkeys(keywords))


# ==============================================================================
# --- 3. Flask 路由與主應用程式邏輯 ---
# 這一區塊是整個應用的進入點和總指揮。
# ==============================================================================


@app.route("/")
def index():
    # 使用 render_template 來渲染 'templates' 資料夾中的 'index.html' 檔案
    return render_template("index.html")


@app.route("/voice")
def voice():
    # 使用 render_template 來渲染 'templates' 資料夾中的 'voice.html' 檔案
    return render_template("voice.html")


# 定義API的路由(URL路徑)和接受的HTTP方法
# 此端點的路徑為 /api/ai-query，且只接受 POST 方法的請求。
@app.route("/api/ai-query", methods=["POST"])
def ai_query():
    """
    處理來自前端的AI查詢請求。
    此函式作為「總指揮」，串連起整個資料處理管道。
    """
    try:
        # --- 步驟 1: 接收並驗證請求 ---
        # 取得前端傳來的 JSON，並由 Flask 自動解析為 Python 字典。
        data = request.get_json()
        # 檢查請求資料是否完整，確保'question'和'keywords'這兩個鍵都存在。
        if not data or "question" not in data or "keywords" not in data:
            return jsonify(
                {"error": "請求格式錯誤，缺少 'question' 或 'keywords' 欄位。"}
            ), 400

        user_question = data["question"]
        original_keywords = data["keywords"]
        print(f"\n收到請求 - 使用者問題: '{user_question}'")

        # --- 步驟 2: 執行資料轉換管道 (Data Pipeline) ---

        # 2a. 從工廠取得所需的 AI 服務物件 (OOP 策略模式)
        # 未來若要切換，只需更改頂部的 AI_SERVICE 常數
        ai_service = get_ai_service(AI_SERVICE)

        # 2b. 呼叫 AI 服務取得解析後的「原始關鍵字列表」 (FP 流程開始)
        llm_answer_keywords = ai_service.get_ai_keywords(
            user_question, original_keywords
        )

        # 2c. 驗證與標準化
        validated_keywords = validate_and_normalize_keywords(
            llm_answer_keywords, original_keywords
        )

        # 2d. 去除重複
        final_keywords = deduplicate_preserve_order(validated_keywords)

        # --- 步驟 3: 輸出除錯資訊並回傳最終結果 ---
        print(f"從AI回應中初步提取的關鍵字: {llm_answer_keywords}")
        print(f"經過驗證後，最終合法的關鍵字: {final_keywords}")
        # 回傳標準化後的結果給前端
        return jsonify({"relevant_keywords": final_keywords})

    # --- 步驟 4: 統一的錯誤處理 ---
    except requests.exceptions.RequestException as e:
        print(f"錯誤: 無法連接至 AI 服務 - {e}")
        return jsonify({"error": f"無法連接至 AI 服務: {e}"}), 503
    except json.JSONDecodeError as e:
        print(f"錯誤: AI 模型回傳格式無法解析 - {e}")
        return jsonify({"error": "AI 模型回傳格式錯誤，無法解析。"}), 500
    except NotImplementedError as e:
        print(f"錯誤: 嘗試呼叫未實作的服務 - {e}")
        return jsonify({"error": str(e)}), 501
    except Exception as e:
        print(f"錯誤: 發生未預期錯誤 - {e}")
        return jsonify({"error": f"伺服器發生未預期錯誤: {e}"}), 500


# ------------------- 啟動伺服器 -------------------
if __name__ == "__main__":
    print("AI查詢服務已啟動，請在瀏覽器中開啟 http://127.0.0.1:5000")
