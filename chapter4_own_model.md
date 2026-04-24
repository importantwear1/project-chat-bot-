# CHAPTER 4 — IMPLEMENTATION DETAILS

---

## 4.1 Deep NLP Pipeline Implementation

### 4.1.1 Overview

The NLP pipeline is the brain of the chatbot. It takes a raw message from the user and converts it into a meaningful, language-appropriate response. This system uses models that we trained ourselves — fine-tuned on Indian language data collected specifically for this project. There is no dependency on any external AI service like ChatGPT or Ollama.

The pipeline runs six stages in order for every message:

```
User Message
     ↓
Stage 1: Language Detection      (FastText — pretrained, fine-tuned if needed)
     ↓
Stage 2: Text Preprocessing      (IndicNLP — normalization + tokenization)
     ↓
Stage 3: Intent Classification   (our fine-tuned MuRIL/mBERT model)
     ↓
Stage 4: Entity Extraction       (our fine-tuned mBERT NER model)
     ↓
Stage 5: Knowledge Base Query    (JSON/PostgreSQL lookup)
     ↓
Stage 6: Response Generation     (template fill + IndicBART fallback)
     ↓
Final Response (in user's language)
```

---

### 4.1.2 Application Startup and Pipeline Initialization

When the FastAPI server starts, it uses Python's `asynccontextmanager` to load all trained models into memory once. Loading happens in six steps:

**Step 1 — Load Intent Classifier:**
Loads our fine-tuned MuRIL model from `models/intent_classifier/` using HuggingFace transformers.
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("models/intent_classifier/")
model = AutoModelForSequenceClassification.from_pretrained("models/intent_classifier/")
```
Console: `Intent classifier loaded: 20 classes, 7 languages`

**Step 2 — Load NER Model:**
Loads our fine-tuned mBERT token classification model from `models/ner_model/`.
Console: `NER model loaded: 8 entity types`

**Step 3 — Load FastText Language Detector:**
Loads `models/language_detection/lid.176.bin`. Falls back to Unicode script detection if missing.
Console: `FastText language detector loaded`

**Step 4 — Load Knowledge Base:**
Reads `data/knowledge_base/entries.json` into memory as a Python dictionary.
Console: `Knowledge Base loaded: 20 intents`

**Step 5 — Load IndicBART Fallback Generator:**
Loads our fine-tuned IndicBART model from `models/indicbart/` for open-ended responses.
Console: `IndicBART fallback generator loaded`

**Step 6 — Initialize Session Store:**
Creates `self.sessions = {}` to hold active conversation sessions.

After all steps: `NLP Pipeline ready!`

---

### 4.1.3 The FastAPI Request-Response Flow

Every message arrives at `/chat/message` as an HTTP POST with three fields:

| Field | Type | Required | Default |
|---|---|---|---|
| `message` | string | Yes | — |
| `session_id` | string | No | auto-generated |
| `user_id` | string | No | "guest" |

**Stage 1 — Pydantic Validation:**
If `message` is empty, Pydantic immediately returns HTTP 422. The pipeline never runs.

**Stage 2 — Session ID Assignment:**
```python
session_id = session_id or str(uuid.uuid4())
```

**Stage 3 — Pipeline Invocation:**
```python
result = pipeline.process(message, session_id)
```

**Stage 4 — Response:**
Returns JSON with: `response`, `intent`, `confidence`, `language`, `session_id`, `entities`, `timestamp`.

---

### 4.1.4 Stage 1 — Language Detection

**Primary: FastText**

FastText represents text as character n-grams (2–5 characters), averages them into a vector, and compares against 176 language vectors. Takes under 1 millisecond.

```python
predictions = self.model.predict(cleaned_text, k=1)
label = predictions[0][0]              # e.g., '__label__hi'
confidence = float(predictions[1][0])  # e.g., 0.97
lang_code = label.replace('__label__', '')
```

If confidence < 0.85, the user is asked to rephrase. Detected language is validated against our supported set: `{hi, bn, mr, ta, te, kn, en}`. Unsupported languages default to English.

**Fallback: Unicode Script Detection**

If FastText model is missing, each character's Unicode code point is checked:

| Language | Unicode Range |
|---|---|
| Hindi/Marathi (Devanagari) | U+0900 – U+097F |
| Bengali | U+0980 – U+09FF |
| Tamil | U+0B80 – U+0BFF |
| Telugu | U+0C00 – U+0C7F |
| Kannada | U+0C80 – U+0CFF |

First Indic character found determines the language. No Indic characters → English.

Target accuracy: ≥95% on test set.

---

### 4.1.5 Stage 2 — Text Preprocessing

**Step 1 — Unicode NFC Normalization:**
```python
import unicodedata
text = unicodedata.normalize('NFC', text)
```
Collapses multiple Unicode encodings of the same character into one canonical form.

**Step 2 — Noise Removal:**
Strips URLs, email addresses, excessive punctuation, and emojis.

**Step 3 — IndicNLP Tokenization:**
Uses language-specific tokenizers from the IndicNLP Library for Indic scripts. English uses BERT WordPiece tokenizer.
```python
from indicnlp.tokenize import indic_tokenize
tokens = indic_tokenize.trivial_tokenize(text, lang='hi')
```

**Step 4 — Lowercase (Latin only):**
Applied only to English input. Indian scripts have no uppercase/lowercase distinction.

---

### 4.1.6 Stage 3 — Intent Classification (Our Trained Model)

**Model: Fine-tuned MuRIL**

MuRIL (Multilingual Representations for Indian Languages) by Google is pretrained on 17 Indian languages including transliterated forms. We add a softmax classification head and fine-tune on our labeled dataset.

**Training Configuration:**

| Parameter | Value |
|---|---|
| Base Model | `google/muril-base-cased` |
| Classification Head | Linear: 768 → 20 classes |
| Training Examples | 24,000 labeled messages (7 languages) |
| Train/Val/Test | 70% / 10% / 20% |
| Optimizer | AdamW, lr = 2e-5 |
| Epochs | 5 |
| Batch Size | 32 |
| Loss | Cross-Entropy |
| Platform | Google Colab Pro (T4 GPU) |

**How It Works:**

The preprocessed text is tokenized and a `[CLS]` token prepended:
```
[CLS] मेरा ऑर्डर कहाँ है [SEP]
```
After 12 transformer encoder layers, the `[CLS]` vector (768-dimensional) passes through the linear head. Softmax converts scores to probabilities:
```
P(intent_k | message) = exp(z_k) / Σ exp(z_j)
```
Highest probability intent above **70% threshold** is selected. Below 70% → clarification prompt. After 2 failed clarifications → escalation/fallback.

**Supported Intents:**

| Intent | English Example | Hindi Example |
|---|---|---|
| `greeting` | "Hello" | "नमस्ते" |
| `track_order` | "Where is my order?" | "मेरा ऑर्डर कहाँ है?" |
| `return_product` | "I want to return this" | "मैं यह वापस करना चाहता हूँ" |
| `payment_issue` | "Payment failed" | "पेमेंट फेल हो गई" |
| `cancel_order` | "Cancel my order" | "मेरा ऑर्डर रद्द करो" |
| `farewell` | "Goodbye" | "अलविदा" |
| `general_faq` | "What are your store hours?" | "दुकान कब खुलती है?" |
| `ask_name` | "What is your name?" | "तुम्हारा नाम क्या है?" |

Target: F1-score ≥ 85% across all 7 languages.

---

### 4.1.7 Stage 4 — Entity Extraction (Our Trained NER Model)

**Model: Fine-tuned mBERT for Token Classification**

Same mBERT base, but produces one label per token using BIO tagging:
- `B-ORDER_ID` — beginning of order ID
- `I-ORDER_ID` — continuation
- `O` — not an entity

**Supported Entity Types:**

| Entity | Example |
|---|---|
| `ORDER_ID` | ORD98765 |
| `PRODUCT_NAME` | "blue kurta", "iPhone" |
| `DATE` | "yesterday", "15 April" |
| `PHONE_NUMBER` | 9876543210 |
| `AMOUNT` | ₹499 |
| `LOCATION` | "Mumbai", "मुंबई" |
| `LANGUAGE_NAME` | "Hindi", "Tamil" |
| `PERSON_NAME` | "Rahul", "Priya" |

**Regex for Structured Entities:**
For order IDs and phone numbers, regex runs alongside the neural model and takes priority:
```python
ORDER_PATTERN = re.compile(r'\b(ORD|#)\d{5,10}\b', re.IGNORECASE)
```

Target: F1-score ≥ 80% on entity test set.

---

### 4.1.8 Stage 5 — Knowledge Base Query

The knowledge base is stored in `data/knowledge_base/entries.json`. Each entry:

```json
{
  "intent": "track_order",
  "dynamic": false,
  "responses": {
    "en": ["Your order {ORDER_ID} is on its way!"],
    "hi": ["आपका ऑर्डर {ORDER_ID} रास्ते में है!"],
    "ta": ["உங்கள் ஆர்டர் {ORDER_ID} வழியில் உள்ளது!"]
  }
}
```

- `dynamic: false` → use pre-written response (fast, reliable)
- `dynamic: true` → use IndicBART generative model

`random.choice()` picks one response from the list for natural variation. Falls back to English if no entry exists for the detected language.

---

### 4.1.9 Stage 6 — Response Generation

**Case 1: Static KB Response (most requests)**
Entity values are substituted into the template:
```python
response = template.replace("{ORDER_ID}", entities.get("ORDER_ID", ""))
# "Your order ORD98765 is on its way!"
```
Takes under 1 millisecond.

**Case 2: IndicBART Fallback (open-ended questions)**
Our fine-tuned IndicBART seq2seq model generates a response:
- Prompt: `[LANG: hi] User: <message>\nBot:`
- Max tokens: 80
- Temperature: 0.7
- Beam search: beam size 4

**Case 3: Translation**
If the KB response exists only in English but the user wrote in Tamil, IndicTrans2 translates it:
```python
translated = indictrans.translate(english_response, src="en", tgt="ta")
```

---

### 4.1.10 Session Management

Sessions stored in `self.sessions` dictionary (in-memory; upgradeable to Redis).

**Update after each message:**
```python
session["history"].append({"role": "user",      "text": user_message})
session["history"].append({"role": "assistant",  "text": bot_reply})
```

**Sliding window — keep last 10 entries (5 exchanges):**
```python
if len(session["history"]) > 10:
    session["history"] = session["history"][-10:]
```

**Entity carryover:** Entities from earlier turns are remembered without the user needing to repeat. Most recent value always takes priority.

---

## 4.2 Voice Interaction Implementation

### 4.2.1 Overview

Voice interaction adds two capabilities: speech input (Speech-to-Text / ASR) and speech output (Text-to-Speech / TTS). The NLP pipeline from Section 4.1 requires no modification — voice is added as a preprocessing and postprocessing wrapper around the existing text pipeline.

```
Audio Input → ASR Service → [Text NLP Pipeline] → TTS Service → Audio Output
```

---

### 4.2.2 Speech-to-Text (ASR) Implementation

**Primary Model: IndicConformer (AI4Bharat)**

IndicConformer is an ASR model purpose-built for Indian languages, trained on the IndicVoices dataset covering 22 Indian languages. It uses a Conformer architecture — a combination of CNN layers (for local acoustic patterns) and self-attention (for long-range context).

We use IndicConformer as our primary ASR model because:
- It is trained specifically on Indian language speech
- It handles accent variation across regions
- It achieves approximately 13.4% WER across 6 Indian languages
- It is freely available from AI4Bharat

**Secondary Model: Whisper (Fine-tuned fallback)**

OpenAI Whisper (medium variant) fine-tuned on Hindi IndicVoices data is kept as a fallback if IndicConformer is unavailable. Whisper achieves approximately 10–12% WER on Hindi but rises to 20–35% for Dravidian languages.

**Audio Capture (Frontend):**
When the user clicks the microphone button, the browser's `MediaDevices.getUserMedia()` API captures audio using `MediaRecorder`. The audio blob (WebM format) is sent to `/voice/transcribe` as a multipart upload.

**Audio Preprocessing (Three Steps):**

1. **Resampling:** All audio converted to 16kHz mono WAV — the standard input format for IndicConformer and Whisper. Different microphones record at different rates (44.1kHz, 48kHz); normalization ensures consistent model performance.

2. **Noise Suppression:** Spectral subtraction estimates background noise from silent portions and subtracts it from speech-active portions. Improves WER by 33–41% in noisy environments.

3. **Silence Trimming:** Leading and trailing silence removed using Voice Activity Detection (VAD). Reduces audio duration sent to model.

**Transcription:**
Preprocessed audio passes to IndicConformer. The transcribed text appears in the chat input box — the user can review and correct it before submitting.

**ASR Target:** WER ≤ 15% for all 6 languages. Expected ranges:
- Hindi, Bengali: 10–12% WER
- Tamil, Telugu: 12–14% WER
- Kannada: 14–17% WER (documented as limitation)

---

### 4.2.3 Text-to-Speech (TTS) Implementation

**Primary: Bhashini TTS API**
Government of India's free TTS service. Good quality for Hindi, Marathi, Bengali, and Tamil. Used for Indo-Aryan languages.

**Secondary: Sarvam AI TTS**
Startup specializing in Indian language voice. Higher naturalness for Dravidian languages (Tamil, Telugu, Kannada). Evaluated for Dravidian language responses.

**Fallback: gTTS (Google Text-to-Speech)**
Used when both primary services are unavailable. Supports all 7 languages. Audio streamed in-memory without saving to disk:

```python
from gtts import gTTS
import io

tts = gTTS(text=response_text, lang=language_code)
audio_buffer = io.BytesIO()
tts.write_to_fp(audio_buffer)
audio_buffer.seek(0)
return StreamingResponse(audio_buffer, media_type="audio/mpeg")
```

**Language-to-TTS Provider Mapping:**

| Language | Primary Provider | Fallback |
|---|---|---|
| Hindi | Bhashini | gTTS |
| Marathi | Bhashini | gTTS |
| Bengali | Bhashini | gTTS |
| Tamil | Sarvam AI | Bhashini → gTTS |
| Telugu | Sarvam AI | Bhashini → gTTS |
| Kannada | Sarvam AI | Bhashini → gTTS |
| English | gTTS | Browser SpeechSynthesis |

---

### 4.2.4 Translation Implementation

When the user asks for a translation (intent: `translation_request`), the pipeline calls `_handle_translation()`.

**Target Language Detection:**
Keywords in the message are scanned for language names:
- "translate to Tamil" → target: `ta`
- "Hindi mein batao" → target: `hi`

**Translation Priority:**
1. **IndicTrans2** (AI4Bharat) — highest quality for Indian language pairs
2. **Bhashini Translation API** — good for Hindi-centric pairs
3. **deep-translator** (Google Translate wrapper) — universal fallback

The translated response is prefixed with 🔄 to visually indicate it is a translation.

---

### 4.2.5 Voice Latency Budget

Target: end-to-end voice response under 3 seconds.

| Stage | Target Time |
|---|---|
| Audio capture + upload | 0.2 – 0.4 s |
| ASR (IndicConformer) | 0.8 – 1.2 s |
| NLP Pipeline (all 6 stages) | 0.3 – 0.8 s |
| TTS synthesis | 0.4 – 0.8 s |
| Audio delivery to browser | 0.1 – 0.2 s |
| **Total** | **1.8 – 3.4 s** |

Pipeline stages are overlapped where possible: NLP begins as soon as ASR finishes; TTS begins as soon as the response text is ready.

---

## 4.3 Hardware and Deployment Framework

### 4.3.1 Development Hardware Requirements

The system is designed to run without a dedicated GPU for inference. Model training uses Google Colab Pro (T4 GPU). Inference runs on CPU.

**Minimum Requirements:**

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores, 2.5 GHz | 8 cores, 3.5 GHz |
| RAM | 8 GB | 16 GB |
| Disk | 15 GB free | 30 GB free |
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 LTS |
| Python | 3.10+ | 3.11 |
| CUDA (training only) | 11.8 | 12.1 |

**Memory Usage Breakdown (Inference, CPU):**

| Component | RAM Usage |
|---|---|
| MuRIL intent classifier | ~1.2 GB |
| mBERT NER model | ~1.2 GB |
| IndicBART fallback | ~1.8 GB |
| FastText model | ~5 MB |
| Knowledge base (20 intents × 7 languages) | ~2 MB |
| Session store (100 concurrent users) | ~50 MB |
| FastAPI server + Python runtime | ~200 MB |
| **Total at full load** | **~4.5 GB** |

Comfortably within 8 GB minimum; ideal on 16 GB systems.

**Inference Speed (Intel Core i7, 8 cores, no GPU):**

| Component | Speed |
|---|---|
| FastText language detection | < 1 ms |
| IndicNLP preprocessing | < 5 ms |
| MuRIL intent classification | 80 – 150 ms |
| mBERT NER extraction | 80 – 150 ms |
| Knowledge base lookup | < 1 ms |
| IndicBART generation (fallback) | 1.5 – 3 s |
| **Total (KB path)** | **~300 – 400 ms** |
| **Total (IndicBART path)** | **~2 – 4 s** |

---

### 4.3.2 Model Training Setup

All models are trained on Google Colab Pro using a T4 GPU (16 GB VRAM). Training is managed with HuggingFace `Trainer` API and experiment results logged with MLflow.

**Intent Classifier Training:**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="models/intent_classifier/",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_f1
)
trainer.train()
```

**Training Time Estimates (T4 GPU):**

| Model | Dataset Size | Training Time |
|---|---|---|
| MuRIL Intent Classifier | 24,000 examples | ~2 hours |
| mBERT NER | 15,000 annotated tokens | ~1.5 hours |
| IndicBART Fallback | 5,000 conversation pairs | ~4 hours |

---

### 4.3.3 Software Stack

**Backend Framework:**
- **FastAPI 0.104+** — main web framework with automatic API docs, Pydantic validation, async support
- **Uvicorn** — ASGI server running FastAPI with asyncio event loop

**AI and NLP Libraries:**
- **HuggingFace Transformers** — loads and runs MuRIL, mBERT, IndicBART models
- **HuggingFace Datasets** — data loading and preprocessing for training
- **IndicNLP Library** — Indian language tokenization and preprocessing
- **FastText** — language identification (Meta pretrained model)
- **IndicTrans2** — Indian language machine translation (AI4Bharat)
- **gTTS** — fallback Text-to-Speech
- **MLflow** — experiment tracking during model training

**Data Storage:**
- **JSON files** — Knowledge base (simple, human-readable, admin-editable)
- **Python dict** — In-memory session store (development)
- **PostgreSQL** — Production: users, sessions, intents, knowledge base
- **MongoDB** — Production: conversation logs and analytics
- **Redis** — Production: session cache for fast multi-turn context

**Frontend:**
- **HTML / CSS / JavaScript** — chat widget (no framework, lightweight and embeddable)
- **React** — admin dashboard
- **EventSource API** — Server-Sent Events for streaming responses

**Infrastructure:**
- **Docker** — each service in its own container
- **Docker Compose** — local development orchestration
- **AWS EC2** — production deployment (g4dn.xlarge for model training; t3.large for inference)
- **AWS S3** — model checkpoints, datasets, audio files
- **GitHub Actions** — CI/CD pipeline with automated test runs

---

### 4.3.4 Docker Deployment Architecture

Each microservice has its own `Dockerfile` and `requirements.txt`. Running `docker-compose up` starts all containers in dependency order:

1. Databases start first (PostgreSQL, MongoDB, Redis)
2. NLP service starts (loads all models)
3. Voice service starts
4. API gateway starts last

**Volume mounts** share the `models/` and `data/` directories across containers — model files are not copied into every image, saving disk space.

**Health Checks:**
Docker checks `/health` every 30 seconds. Three consecutive failures trigger automatic container restart. Console example output on startup:
```
✅ Intent classifier loaded: 20 classes, 7 languages
✅ NER model loaded: 8 entity types
✅ FastText language detector loaded
✅ Knowledge Base loaded: 20 intents
✅ IndicBART fallback generator loaded
✅ NLP Pipeline ready!
```

---

### 4.3.5 REST API Endpoint Summary

**Chat Endpoints:**

| Method | URL | Purpose |
|---|---|---|
| POST | /chat/message | Process message, return full response |
| POST | /chat/stream | Stream response token by token (SSE) |
| POST | /chat/session/start | Create new session |
| POST | /chat/session/end | End and clear session |

**Voice Endpoints:**

| Method | URL | Purpose |
|---|---|---|
| POST | /voice/transcribe | Convert uploaded audio to text (ASR) |
| POST | /voice/synthesize | Convert text to MP3 audio (TTS) |

**Translation and Detection:**

| Method | URL | Purpose |
|---|---|---|
| POST | /translate | Translate text between Indian languages |
| POST | /detect-language | Identify language of input text |

**Admin Endpoints:**

| Method | URL | Purpose |
|---|---|---|
| GET | /admin/knowledge-base | List all KB intents |
| POST | /admin/knowledge-base | Add or update an intent |
| DELETE | /admin/knowledge-base/{intent} | Delete an intent |
| GET | /admin/analytics | Session and message statistics |
| GET | /admin/sessions | List all active sessions |

**System:**

| Method | URL | Purpose |
|---|---|---|
| GET | /health | Check all services running |
| GET | /languages | List supported languages |

---

### 4.3.6 Error Handling and Fallback Strategy

The system stays functional even when individual components fail:

**If MuRIL intent model fails to load:**
The system returns knowledge base responses using regex pattern matching only. A warning is printed but the server stays running.

**If FastText model is missing:**
Unicode script-based detection takes over. Language detection continues with slightly lower accuracy for distinguishing Hindi from Marathi.

**If IndicBART is unavailable:**
Dynamic intents return a polite message: "I am unable to answer that question right now. Please contact support." Static KB intents are unaffected.

**If knowledge base file is missing:**
Server starts with empty KB. All requests fall through to IndicBART. Console shows warning.

**If TTS fails (Bhashini/Sarvam API error):**
System automatically tries next provider in chain: Bhashini → Sarvam AI → gTTS. If all fail, the text response is still shown to the user with a visual speaker-unavailable icon.

**If session ID is not provided:**
New UUID generated automatically. Conversation starts fresh.

Every endpoint is wrapped in try-except blocks. Unhandled exceptions return HTTP 500 with a descriptive error message. The server never crashes on a single bad request.

---

### 4.3.7 Security Considerations

**CORS Configuration:**
Development allows all origins (`allow_origins=["*"]`). Production restricts to deployed domain only.

**Input Validation:**
All fields validated by Pydantic. `message` has minimum length 1. All field types are constrained. Invalid requests never reach the pipeline.

**No Authentication (Current Version):**
Admin endpoints are currently open. For production, the API Gateway's `auth.py` middleware activates JWT token validation for all admin access.

**Data Privacy:**
Conversation history stored only in memory — never written to disk. All history permanently deleted when session ends or server restarts. Raw audio files are discarded immediately after transcription. No personal data persists beyond the session lifetime. Designed to comply with India's Digital Personal Data Protection Act 2023.

**Rate Limiting:**
Nginx is configured with rate limiting: 60 requests/minute per IP for text endpoints, 10 requests/minute per IP for voice endpoints (voice processing is compute-intensive).

---

*End of Chapter 4*
