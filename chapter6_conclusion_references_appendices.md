# CHAPTER 6 — CONCLUSION & FUTURE SCOPE

---

## 6.1 Conclusion

This project presents the design, development, and empirical evaluation of a **Multilingual AI Chatbot with Voice Interaction and Context Awareness**, purpose-built for the Indian e-commerce domain. The system addresses a critical gap in conversational AI deployment in India — the inability of existing commercial chatbots to handle the country's rich linguistic diversity with satisfactory accuracy, natural voice interaction, and persistent session context.

### 6.1.1 Summary of Work Accomplished

The project delivered a fully functional, end-to-end conversational AI system with the following key contributions:

| Component | What Was Built | Key Result |
|---|---|---|
| **Intent Classification** | Fine-tuned MuRIL on 28,568 multilingual utterances across 7 languages | Macro F1 = 90.2 % (target ≥ 85 %) |
| **Language Detection** | FastText-based lid.176 model with heuristic override | Accuracy = 97.1 % (target ≥ 95 %) |
| **Voice Input (ASR)** | IndicConformer deployed via REST endpoint | Avg WER = 14.8 % clean (target ≤ 15 %) |
| **Response Generation** | Hybrid KB-path + Ollama LLM fallback pipeline | KB latency = 312 ms avg |
| **Text-to-Speech (TTS)** | Coqui XTTS-v2 for Indic languages | End-to-end voice latency = 2.8 s avg |
| **Context Management** | In-memory session store with 30-min TTL | 100 % session retention within TTL |
| **Backend API** | FastAPI with 8 endpoints, Pydantic validation | 10/10 flow tests passed; 0 % error at 50 users |
| **Frontend** | Responsive HTML/CSS/JS chat UI with dark mode | CSAT = 4.28 / 5.00 (85.6 %) |

### 6.1.2 Key Findings

**1. MuRIL outperforms general multilingual models for Indic NLP.**
The fine-tuned MuRIL model achieved consistent superiority over comparable multilingual baselines (mBERT, XLM-R) across all six Indic languages. Its pre-training on transliterated and code-switched Indian text gave it a decisive advantage in capturing morphological nuances specific to Hindi, Marathi, Bengali, Tamil, Telugu, and Kannada.

**2. The `return_product` ↔ `cancel_order` intent pair is the universal bottleneck.**
Across all languages, this single conflict pair accounts for the majority of misclassifications. It is rooted in semantic overlap that cannot be resolved by the language model alone — it requires downstream entity-aware logic (e.g., checking shipment status before committing to an intent label). This finding has practical implications for any e-commerce chatbot deployment.

**3. Dravidian languages (Tamil, Telugu, Kannada) require dedicated architectural attention.**
Due to SOV syntax, heavy agglutination, and lower pre-training corpus coverage in MuRIL, these three languages exhibited consistently higher error rates than Indo-Aryan languages. Future work must include language-family-specific data augmentation pipelines.

**4. IndicConformer decisively outperforms Whisper-medium for Indian ASR.**
Whisper-medium's WER on Kannada (28.6 %) was more than double IndicConformer's (13.9 %). This validates the use of an India-specific ASR architecture over general-purpose multilingual models for production Indic voice interfaces.

**5. The hybrid KB-path + LLM architecture balances speed and generality.**
The knowledge-base-first routing strategy reduces average response latency for routine queries (track, cancel, return, payment) to under 350 ms, while the Ollama IndicBART fallback provides coverage for open-ended or unexpected queries. This architectural pattern is recommended for any domain-specific chatbot where latency is a primary constraint.

**6. All eight project targets were met or exceeded.**
The system exceeded every quantitative target set at project inception — including intent F1, WER, pipeline latency, voice latency, CSAT score, and concurrent user capacity. This validates the overall architectural and implementation choices made throughout the project.

### 6.1.3 Academic and Industrial Significance

From an academic perspective, this project demonstrates that a resource-constrained team (no GPU server, standard consumer hardware) can fine-tune and deploy a state-of-the-art multilingual NLP system by making careful model selection decisions. The use of MuRIL, IndicConformer, and IndicBART — all open-source AI4Bharat models — as foundational components provides a replicable blueprint for future Indic NLP researchers.

From an industrial perspective, the system demonstrates that a multilingual Indian e-commerce chatbot capable of handling 50 concurrent users at sub-second latency is achievable without proprietary cloud NLP APIs. This has direct implications for cost-effective deployment by small and medium e-commerce enterprises operating in vernacular Indian markets.

---

## 6.2 Future Scope

While the current system meets all project objectives, several directions for improvement and extension have been identified during development and testing.

### 6.2.1 Short-Term Enhancements (0–6 Months)

| Enhancement | Description | Expected Impact |
|---|---|---|
| **Redis Session Store** | Replace in-memory dict with Redis for persistent, distributed sessions | Eliminates 2.1 % error at 100 users; enables horizontal scaling |
| **Entity-Aware Intent Override** | Check shipment status API before finalising `return` vs `cancel` label | Eliminates the #1 error source across all languages |
| **Kannada-Specific Augmentation** | Generate 2,000 additional Kannada training samples via back-translation | Expected F1 improvement: +2 to +4 % |
| **Noise Suppression Upgrade** | Replace spectral subtraction with RNNoise or DeepFilterNet | Expected WER reduction in noisy conditions: 5–10 % |
| **UI Enhancements** | Add typing indicators, read receipts, voice visualiser waveform | Improved user engagement and CSAT |

### 6.2.2 Medium-Term Research Directions (6–18 Months)

**1. Dialect and Code-Switching Support**
Indian users frequently mix languages within a single sentence (e.g., "Mera order kab aayega bhai?" — Hindi + English). The current system handles language alternation at the turn level but not within a single utterance. Future work should include code-switch-aware tokenisation and training data that mirrors real-world Hinglish, Tanglish, and Manglish usage patterns.

**2. Personalised Response Generation**
The current system uses a single LLM prompt template for all users. Incorporating a user preference profile — preferred language, formality level, prior purchase history — into the Ollama prompt context would enable personalised responses. This aligns with the emerging field of personalised dialogue systems.

**3. Multimodal Input Support**
Extending the system to accept image inputs (e.g., "Here is a photo of the damaged product") using a Vision-Language Model (VLM) such as LLaVA would significantly expand the chatbot's capability for complaint and return workflows where visual evidence is standard practice.

**4. Federated Learning for Privacy-Preserving Improvement**
As the system collects interaction logs, future model updates should be performed using Federated Learning — training on decentralised user devices without transmitting raw conversation data to a central server. This is particularly important for Indian users given growing data privacy awareness.

**5. Adaptation to Additional Indic Languages**
The current system covers 6 Indic languages. India has 22 scheduled languages. Future iterations should extend coverage to Odia, Punjabi, Gujarati, Assamese, and Malayalam using the same MuRIL fine-tuning pipeline with language-specific data collection sprints.

### 6.2.3 Long-Term Vision (18+ Months)

**1. Full Agentic Chatbot with Tool Use**
Future versions could integrate tool-use capabilities, allowing the chatbot to directly call backend APIs (order management system, payment gateway, inventory database) and execute transactions on behalf of the user rather than merely providing information. This would transform the chatbot from a passive Q&A system to an active e-commerce agent.

**2. Deployment on Edge Devices**
Quantised versions of MuRIL (INT8/INT4 via ONNX Runtime) and IndicConformer could be deployed on mobile devices for offline, privacy-first operation. This is crucial for users in rural India with intermittent internet connectivity.

**3. Autonomous Continuous Learning Pipeline**
A production pipeline where high-confidence conversation logs are automatically sampled, human-reviewed, labelled, and fed back into weekly fine-tuning cycles would enable the model to continuously improve without manual intervention.

---

*End of Chapter 6*

---
---

# CHAPTER 7 — REFERENCES

> All references are formatted per **IEEE Citation Style**.

---

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, Minneapolis, MN, USA, Jun. 2019, pp. 4171–4186.

[3] K. Khanuja, D. Bansal, S. Mehtani, S. Khosla, A. Dey, B. Gopalan, D. K. Margam, P. Aggarwal, R. T. Nagipogu, S. Dave, S. Gupta, S. B. Chandra Gali, U. Subramanian, and P. Talukdar, "MuRIL: Multilingual representations for Indian languages," *arXiv preprint arXiv:2103.10730*, 2021.

[4] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 12449–12460.

[5] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, P. Copet, G. Hsu, A. Baevski, et al., "Scaling speech technology to 1,000+ languages," *arXiv preprint arXiv:2208.01618*, 2022.

[6] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, "Unsupervised cross-lingual representation learning at scale," in *Proc. ACL*, Online, Jul. 2020, pp. 8440–8451.

[7] A. Radford, J. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. ICML*, Honolulu, HI, USA, Jul. 2023, pp. 28492–28518.

[8] AI4Bharat, "IndicConformer: A state-of-the-art ASR model for Indian languages," *GitHub repository*, 2023. [Online]. Available: https://github.com/AI4Bharat/IndicConformer

[9] AI4Bharat, "IndicTrans2: Towards high-quality and responsible development of machine translation systems for all 22 scheduled Indian languages," *arXiv preprint arXiv:2305.16307*, 2023.

[10] V. Dabre, A. Kunchukuttan, P. Bhatt, S. Puduppully, A. Kumar, A. Murthy, M. Richardson, and P. Bhattacharyya, "IndicBART: A pre-trained model for natural language generation of Indic languages," *arXiv preprint arXiv:2109.02903*, 2021.

[11] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

[12] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv preprint arXiv:1907.11692*, 2019.

[13] T. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, "Enriching word vectors with subword information," *Transactions of the Association for Computational Linguistics*, vol. 5, pp. 135–146, 2017.

[14] A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, "Bag of tricks for efficient text classification," in *Proc. EACL*, Valencia, Spain, Apr. 2017, pp. 427–431.

[15] A. Défossez, G. Synnaeve, and Y. Adi, "Real time speech enhancement in the waveform domain," in *Proc. Interspeech*, Brno, Czech Republic, 2020, pp. 3291–3295.

[16] S. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, "SQuAD: 100,000+ questions for machine comprehension of text," in *Proc. EMNLP*, Austin, TX, USA, Nov. 2016, pp. 2383–2392.

[17] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al., "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP (System Demonstrations)*, Online, Nov. 2020, pp. 38–45.

[18] S. Ābols, "A survey of voice user interface design," *International Journal of Advanced Computer Science and Applications*, vol. 12, no. 3, pp. 512–521, 2021.

[19] S. Poria, E. Cambria, R. Bajpai, and A. Hussain, "A review of affective computing: From unimodal analysis to multimodal fusion," *Information Fusion*, vol. 37, pp. 98–125, Sep. 2017.

[20] G. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter," *arXiv preprint arXiv:1910.01108*, 2019.

---
---

# APPENDIX A — DETAILED API SPECIFICATIONS

> **Scope:** All REST endpoints exposed by the FastAPI backend at `http://localhost:8000`.

## A.1 Endpoint Summary Table

| # | Method | Endpoint | Description | Auth |
|---|---|---|---|---|
| 1 | POST | `/chat/message` | Send a text message; returns intent, response, language | None |
| 2 | POST | `/voice/transcribe` | Upload audio; returns transcript + NLP response | None |
| 3 | GET | `/session/{session_id}` | Retrieve conversation history for a session | None |
| 4 | DELETE | `/session/{session_id}` | Clear session context | None |
| 5 | POST | `/tts/synthesise` | Convert text to speech; returns WAV audio stream | None |
| 6 | GET | `/health` | Liveness probe; returns server status | None |
| 7 | GET | `/languages` | List all supported language codes | None |
| 8 | GET | `/intents` | List all supported intent labels | None |

## A.2 Key Endpoint: `POST /chat/message`

**Request Body (JSON):**
```json
{
  "session_id": "abc-123",
  "message":    "मेरा ऑर्डर कहाँ है?",
  "language":   "hi"
}
```

**Response Body (JSON):**
```json
{
  "session_id":    "abc-123",
  "detected_lang": "hi",
  "intent":        "track_order",
  "confidence":    0.94,
  "response":      "आपका ऑर्डर #ORD-7821 रास्ते में है और कल तक पहुँच जाएगा।",
  "latency_ms":    312
}
```

**HTTP Status Codes:** `200 OK` | `422 Unprocessable Entity` (validation error) | `500 Internal Server Error`

## A.3 Key Endpoint: `POST /voice/transcribe`

**Request:** `multipart/form-data` with fields `audio_file` (WAV/MP3/OGG, max 10 MB) and `session_id` (string).

**Response Body (JSON):**
```json
{
  "transcript":    "मेरा ऑर्डर कहाँ है",
  "detected_lang": "hi",
  "intent":        "track_order",
  "confidence":    0.91,
  "response":      "आपका ऑर्डर #ORD-7821 रास्ते में है।",
  "asr_wer_est":   null,
  "latency_ms":    1240
}
```

---

# APPENDIX B — SYSTEM DATABASE SCHEMAS

> **Scope:** All data structures used by the chatbot backend. The current implementation uses in-memory Python dictionaries; schemas below represent the equivalent SQL/NoSQL structure for production deployment.

## B.1 Session Store Schema (Redis / PostgreSQL)

```sql
CREATE TABLE sessions (
    session_id      VARCHAR(36)  PRIMARY KEY,
    user_id         VARCHAR(64),
    language        CHAR(2)      NOT NULL DEFAULT 'en',
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    last_active     TIMESTAMP    NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMP    NOT NULL,
    context_json    JSONB        NOT NULL DEFAULT '{}'
);

-- context_json structure:
-- {
--   "history": [
--     {"role": "user",      "text": "...", "intent": "...", "ts": "..."},
--     {"role": "assistant", "text": "...", "ts": "..."}
--   ],
--   "last_intent": "track_order",
--   "entities": {"order_id": "ORD-7821"}
-- }
```

## B.2 Intent Knowledge Base Schema

```sql
CREATE TABLE intent_responses (
    id          SERIAL       PRIMARY KEY,
    intent      VARCHAR(64)  NOT NULL,
    language    CHAR(2)      NOT NULL,
    template    TEXT         NOT NULL,
    priority    SMALLINT     NOT NULL DEFAULT 0,
    created_at  TIMESTAMP    DEFAULT NOW(),
    UNIQUE (intent, language, priority)
);

CREATE INDEX idx_intent_lang ON intent_responses(intent, language);
```

## B.3 Interaction Log Schema (Analytics)

```sql
CREATE TABLE interaction_logs (
    log_id          BIGSERIAL    PRIMARY KEY,
    session_id      VARCHAR(36)  REFERENCES sessions(session_id),
    turn_number     SMALLINT     NOT NULL,
    user_message    TEXT,
    detected_lang   CHAR(2),
    intent          VARCHAR(64),
    confidence      FLOAT,
    bot_response    TEXT,
    latency_ms      INTEGER,
    asr_used        BOOLEAN      DEFAULT FALSE,
    tts_used        BOOLEAN      DEFAULT FALSE,
    logged_at       TIMESTAMP    DEFAULT NOW()
);
```

---

# APPENDIX C — TEST CASE SUITES

> **Scope:** Selected representative test cases from the full pytest suite. The complete suite contains **47 unit tests** and **10 end-to-end flow tests**.

## C.1 Unit Test Cases (Selected)

| Test ID | Module | Description | Input | Expected Output | Status |
|---|---|---|---|---|---|
| UT-01 | `language_detector` | Detect Hindi | `"नमस्ते"` | `lang = "hi"`, confidence ≥ 0.95 | ✅ Pass |
| UT-02 | `language_detector` | Detect Tamil | `"வணக்கம்"` | `lang = "ta"`, confidence ≥ 0.95 | ✅ Pass |
| UT-03 | `intent_classifier` | Classify track_order (Hindi) | `"मेरा ऑर्डर कहाँ है"` | `intent = "track_order"` | ✅ Pass |
| UT-04 | `intent_classifier` | Classify greeting (English) | `"Hello there"` | `intent = "greeting"` | ✅ Pass |
| UT-05 | `intent_classifier` | Low-confidence fallback | `"???"` | `confidence < 0.5`, fallback to LLM | ✅ Pass |
| UT-06 | `session_manager` | Create new session | `session_id = "test-001"` | Session created, TTL = 1800 s | ✅ Pass |
| UT-07 | `session_manager` | Retrieve existing session | Valid `session_id` | Returns context dict | ✅ Pass |
| UT-08 | `session_manager` | Expired session returns None | TTL-expired session_id | `None` returned | ✅ Pass |
| UT-09 | `nlp_pipeline` | KB path returns in < 500 ms | `"track_order"` intent | `latency_ms < 500` | ✅ Pass |
| UT-10 | `api /health` | Health endpoint responds | GET `/health` | `{"status": "ok"}`, HTTP 200 | ✅ Pass |

## C.2 End-to-End Flow Test Cases

| Flow ID | Language | Conversation Summary | Turns | Result |
|---|---|---|---|---|
| FT-01 | Hindi | Greet → Track Order → Farewell | 3 | ✅ Pass |
| FT-02 | Marathi | Payment Issue → Complaint → Escalate | 3 | ✅ Pass |
| FT-03 | Bengali | Product Inquiry → Cancel → Confirm | 3 | ✅ Pass |
| FT-04 | Tamil | Return (no Order ID) → Clarify → Resolve | 4 | ✅ Pass |
| FT-05 | Telugu | FAQ → Translation Request → Farewell | 3 | ✅ Pass |
| FT-06 | Kannada | Voice Input → Intent → TTS Response | 1 | ✅ Pass |
| FT-07 | Hindi | Ambiguous Message → Low Confidence → Reprompt | 2 | ✅ Pass |
| FT-08 | Mixed | Unknown Language → Fallback English | 1 | ✅ Pass |
| FT-09 | English | Session Timeout → New Session Auto-Create | 2 | ✅ Pass |
| FT-10 | English | Empty Message → HTTP 422 | 1 | ✅ Pass |

**Total: 47 unit tests + 10 flow tests = 57 tests. All 57 passed.**

---

# APPENDIX D — SOURCE CODE APPENDICES

> **Scope:** Condensed source code listings of the most critical modules. Full source code is available in the project repository at `e:\MAAM\chatbot code\`.

## D.1 NLP Pipeline Core (`nlp_pipeline.py` — Key Methods)

```python
class NLPPipeline:
    """Single shared instance; loaded once at FastAPI startup."""

    def __init__(self):
        self.lang_detector   = LanguageDetector()        # FastText lid.176
        self.intent_model    = IntentClassifier()        # Fine-tuned MuRIL
        self.kb              = KnowledgeBase()           # JSON response store
        self.session_manager = SessionManager(ttl=1800)  # In-memory sessions
        self.ollama_client   = OllamaClient(model="indicbart") # LLM fallback

    def process(self, session_id: str, message: str) -> dict:
        lang    = self.lang_detector.detect(message)
        intent, conf = self.intent_model.classify(message, lang)
        history = self.session_manager.get_history(session_id)

        if conf >= 0.70 and self.kb.has_response(intent, lang):
            response = self.kb.get_response(intent, lang, history)
            path = "kb"
        else:
            response = self.ollama_client.generate(message, lang, history)
            path = "llm"

        self.session_manager.append(session_id, message, intent, response)
        return {"intent": intent, "confidence": conf,
                "response": response, "path": path, "lang": lang}
```

## D.2 Intent Classifier (`intent_classifier.py` — Key Methods)

```python
class IntentClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("checkpoints/muril-finetuned")
        self.model     = AutoModelForSequenceClassification.from_pretrained(
                             "checkpoints/muril-finetuned")
        self.model.eval()

    @torch.no_grad()
    def classify(self, text: str, lang: str) -> tuple[str, float]:
        inputs  = self.tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=128)
        logits  = self.model(**inputs).logits
        probs   = torch.softmax(logits, dim=-1)
        top_idx = probs.argmax().item()
        return self.model.config.id2label[top_idx], probs[0][top_idx].item()
```

## D.3 FastAPI App Entry Point (`main.py` — Startup & Route Registration)

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pipeline import NLPPipeline

pipeline: NLPPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = NLPPipeline()   # All models loaded here once
    print("[STARTUP] NLPPipeline ready.")
    yield
    print("[SHUTDOWN] Cleanup complete.")

app = FastAPI(title="Multilingual Chatbot API", lifespan=lifespan)

from routers import chat, voice, session, tts, health
app.include_router(chat.router,    prefix="/chat")
app.include_router(voice.router,   prefix="/voice")
app.include_router(session.router, prefix="/session")
app.include_router(tts.router,     prefix="/tts")
app.include_router(health.router)
```

## D.4 MuRIL Fine-Tuning Script (`train_muril.py` — Training Configuration)

```python
training_args = TrainingArguments(
    output_dir              = "checkpoints/muril-finetuned",
    num_train_epochs        = 5,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_ratio            = 0.1,
    lr_scheduler_type       = "cosine",
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1",
    fp16                    = False,   # CPU training; set True for GPU
    logging_steps           = 50,
    report_to               = "none",
)
```

---

*End of Appendices — End of Report*
