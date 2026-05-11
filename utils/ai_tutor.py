"""
utils/ai_tutor.py
─────────────────
Emotion-adaptive AI teaching responses.
Supports multiple LLM providers — switch with one line in .env

Supported providers (set LLM_PROVIDER in .env):
  groq       → Free, fastest. Llama 4 / DeepSeek-R1 / Mixtral   ← BEST FREE OPTION
  deepseek   → Free generous tier. DeepSeek-V3 / R1
  gemini     → Free tier. Gemini 1.5 Flash / Pro
  openrouter → Free credits. Access to 100+ models
  ollama     → 100% free, runs locally, no internet needed
  anthropic  → Paid. Claude Sonnet (original)

.env example:
  LLM_PROVIDER=groq
  LLM_API_KEY=your_groq_key_here
  LLM_MODEL=llama-4-scout-17b-16e-instruct    # optional override
"""

import os
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Teaching strategy per learning state ──────────────────────────────────────
STRATEGIES = {
    "engaged":    {"tone": "enthusiastic and challenging",
                   "approach": "Push deeper — give harder problems, celebrate curiosity."},
    "confused":   {"tone": "clear, simple and patient",
                   "approach": "Break into tiny steps. Use analogies. Check understanding."},
    "frustrated": {"tone": "calm, empathetic and reassuring",
                   "approach": "Validate effort. Simplify drastically. One step at a time."},
    "disengaged": {"tone": "energetic and stimulating",
                   "approach": "Share a surprising fact. Ask an open question. Keep it short."},
    "anxious":    {"tone": "gentle, warm and confidence-building",
                   "approach": "Normalize mistakes. Start from what they know. Build slowly."},
    "curious":    {"tone": "exploratory and exciting",
                   "approach": "Feed curiosity. Explore tangents. Ask 'what if' questions."},
    "neutral":    {"tone": "friendly and engaging",
                   "approach": "Balanced explanation with a guiding question at the end."},
    "averse":     {"tone": "understanding and reframing",
                   "approach": "Acknowledge feelings. Show a new angle. Make it relevant."},
}

SYSTEM_PROMPT = """You are EVA (Emotion-adaptive Virtual Assistant), an intelligent teaching assistant.

## Student's current emotional state
- Emotion        : {emotion}
- Learning state : {learning_state}
- Energy level   : {energy_level}

## Your teaching strategy for this state
- Tone     : {tone}
- Approach : {approach}

## Subject: {subject}

## Rules
1. Respond naturally to the student's emotion — don't be robotic about it.
2. Keep responses concise (2-3 short paragraphs max).
3. End with ONE guiding question or next step.
4. Never say "As an AI" or "I'm just a program".
5. Adapt vocabulary complexity to their state (simpler when confused/frustrated).
6. Be warm and encouraging regardless of state.
"""

DIFFICULTY_PROMPTS = {
    "easy":   "Use very simple language and basic examples.",
    "medium": "Use clear explanations with relatable examples.",
    "hard":   "Use precise language and challenge with edge cases.",
}


# ── Provider implementations ──────────────────────────────────────────────────

class BaseProvider:
    def chat(self, system: str, messages: List[Dict], max_tokens: int = 800) -> str:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return False

    def name(self) -> str:
        return "base"


class GroqProvider(BaseProvider):
    """
    Free & fastest provider. Sign up at https://console.groq.com
    Recommended free models:
      - llama-4-scout-17b-16e-instruct  (latest Llama 4)
      - deepseek-r1-distill-llama-70b   (DeepSeek R1 distilled)
      - mixtral-8x7b-32768              (Mixtral, very capable)
      - llama3-70b-8192                 (Llama 3 70B)
    """
    DEFAULT_MODEL = "llama-4-scout-17b-16e-instruct"

    def __init__(self, api_key: str, model: str = None):
        self._model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"✅ Groq ready — model: {self._model}")
        except ImportError:
            logger.warning("⚠️  groq not installed: pip install groq")
            self._client = None

    def chat(self, system: str, messages: List[Dict], max_tokens: int = 800) -> str:
        if not self._client:
            return ""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    def is_ready(self) -> bool:
        return self._client is not None

    def name(self) -> str:
        return f"groq/{self._model}"


class DeepSeekProvider(BaseProvider):
    """
    Free generous tier. Sign up at https://platform.deepseek.com
    Models: deepseek-chat (V3), deepseek-reasoner (R1)
    """
    DEFAULT_MODEL = "deepseek-chat"
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str, model: str = None):
        self._model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self._api_key = api_key
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
            logger.info(f"✅ DeepSeek ready — model: {self._model}")
        except ImportError:
            logger.warning("⚠️  openai not installed: pip install openai")
            self._client = None

    def chat(self, system: str, messages: List[Dict], max_tokens: int = 800) -> str:
        if not self._client:
            return ""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    def is_ready(self) -> bool:
        return self._client is not None

    def name(self) -> str:
        return f"deepseek/{self._model}"


class GeminiProvider(BaseProvider):
    """
    Free tier. Sign up at https://aistudio.google.com
    Models: gemini-1.5-flash (free), gemini-1.5-pro (limited free)
    """
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: str, model: str = None):
        self._model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self._model)
            logger.info(f"✅ Gemini ready — model: {self._model}")
        except ImportError:
            logger.warning("⚠️  google-generativeai not installed: pip install google-generativeai")
            self._client = None

    def chat(self, system: str, messages: List[Dict], max_tokens: int = 800) -> str:
        if not self._client:
            return ""
        # Gemini combines system + first user message
        history = []
        for m in messages[:-1]:
            history.append({"role": m["role"], "parts": [m["content"]]})
        last = messages[-1]["content"] if messages else ""
        full_prompt = f"{system}\n\n{last}"
        resp = self._client.generate_content(
            full_prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.7}
        )
        return resp.text.strip()

    def is_ready(self) -> bool:
        return self._client is not None

    def name(self) -> str:
        return f"gemini/{self._model}"


class OpenRouterProvider(BaseProvider):
    """
    Free credits on sign-up. Access to 100+ models including free ones.
    Sign up at https://openrouter.ai
    Free models: meta-llama/llama-4-scout, deepseek/deepseek-r1, mistralai/mistral-7b
    """
    DEFAULT_MODEL = "meta-llama/llama-4-scout"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str = None):
        self._model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
            logger.info(f"✅ OpenRouter ready — model: {self._model}")
        except ImportError:
            logger.warning("⚠️  openai not installed: pip install openai")
            self._client = None

    def chat(self, system: str, messages: List[Dict], max_tokens: int = 800) -> str:
        if not self._client:
            return ""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def is_ready(self) -> bool:
        return self._client is not None

    def name(self) -> str:
        return f"openrouter/{self._model}"


class OllamaProvider(BaseProvider):
    """
    100% free — runs locally on your machine. No internet after setup.
    Install Ollama: https://ollama.com
    Then run: ollama pull llama3 (or deepseek-r1, mistral, phi3, etc.)
    """
    DEFAULT_MODEL = "llama3"
    DEFAULT_URL   = "http://localhost:11434"

    def __init__(self, model: str = None, base_url: str = None):
        self._model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_URL)
        try:
            import requests
            # Quick health check
            r = requests.get(f"{self._base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                self._ready = True
                logger.info(f"✅ Ollama ready — model: {self._model} at {self._base_url}")
            else:
                self._ready = False
                logger.warning("⚠️  Ollama server not running. Start with: ollama serve")
        except Exception:
            self._ready = False
            logger.warning("⚠️  Ollama not reachable. Install from https://ollama.com")

    def chat(self, system: str, messages: List[Dict], max_tokens: int = 800) -> str:
        if not self._ready:
            return ""
        import requests
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7}
        }
        resp = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def is_ready(self) -> bool:
        return self._ready

    def name(self) -> str:
        return f"ollama/{self._model}"


class AnthropicProvider(BaseProvider):
    """Paid — Claude Sonnet via Anthropic API."""
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str, model: str = None):
        self._model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"✅ Anthropic ready — model: {self._model}")
        except ImportError:
            logger.warning("⚠️  anthropic not installed: pip install anthropic")
            self._client = None

    def chat(self, system: str, messages: List[Dict], max_tokens: int = 800) -> str:
        if not self._client:
            return ""
        resp = self._client.messages.create(
            model=self._model, system=system,
            messages=messages, max_tokens=max_tokens,
        )
        return resp.content[0].text.strip()

    def is_ready(self) -> bool:
        return self._client is not None

    def name(self) -> str:
        return f"anthropic/{self._model}"


# ── Provider factory ──────────────────────────────────────────────────────────

def _build_provider() -> BaseProvider:
    """
    Read LLM_PROVIDER from environment and return the right provider instance.
    Falls back to mock responses if nothing is configured.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    api_key  = os.getenv("LLM_API_KEY", "").strip()
    model    = os.getenv("LLM_MODEL", "").strip() or None

    if provider == "groq":
        return GroqProvider(api_key, model)
    elif provider == "deepseek":
        return DeepSeekProvider(api_key, model)
    elif provider == "gemini":
        return GeminiProvider(api_key, model)
    elif provider == "openrouter":
        return OpenRouterProvider(api_key, model)
    elif provider == "ollama":
        return OllamaProvider(model)
    elif provider == "anthropic":
        return AnthropicProvider(api_key, model)
    else:
        logger.warning(f"⚠️  Unknown LLM_PROVIDER='{provider}'. Using mock responses.")
        return BaseProvider()


# ── Mock responses (when no provider is configured) ───────────────────────────

MOCK_RESPONSES = {
    "confused":    "No worries at all! Let's slow down and try a different approach. 😊 Which part is tripping you up the most?",
    "frustrated":  "I hear you — this is genuinely tricky. Let's break it into the smallest possible steps. Take a breath, we'll get through it together. 🤝",
    "engaged":     "Love the energy! 🚀 You're asking exactly the right questions. Let's push even deeper — here's the next level...",
    "disengaged":  "Here's something that might surprise you about this topic... 🌟 Did you know it connects directly to everyday life?",
    "anxious":     "You're doing better than you think. 😊 Mistakes are just data — they show us where to focus. Let's go step by step.",
    "curious":     "Great question — let's go down that rabbit hole! 🔍 This connects to something really fascinating...",
    "neutral":     "Good question! Here's how I'd think about it — let me walk you through the core idea clearly.",
}


# ── Main AITutor class ────────────────────────────────────────────────────────

class AITutor:
    def __init__(self):
        self._provider = _build_provider()
        logger.info(f"🤖 AI Tutor using provider: {self._provider.name()}")

    def get_adaptive_response(
        self,
        student_message: str,
        emotion_state: Dict,
        subject: str = "general",
        conversation_history: List[Dict] = None,
        difficulty: str = "medium",
    ) -> Dict:
        """Generate an emotion-adaptive teaching response."""

        emotion       = emotion_state.get("dominant_emotion", "neutral")
        learning_state = emotion_state.get("learning_state", "neutral")
        energy_level  = emotion_state.get("energy_level", "medium")

        strategy = STRATEGIES.get(learning_state, STRATEGIES["neutral"])

        system = SYSTEM_PROMPT.format(
            emotion=emotion,
            learning_state=learning_state,
            energy_level=energy_level,
            tone=strategy["tone"],
            approach=strategy["approach"],
            subject=subject,
        ) + f"\nDifficulty guidance: {DIFFICULTY_PROMPTS.get(difficulty, '')}"

        # Build message list (last 10 turns)
        messages = list(conversation_history or [])[-10:]
        messages.append({"role": "user", "content": student_message})

        if not self._provider.is_ready():
            return {
                "message": MOCK_RESPONSES.get(learning_state, MOCK_RESPONSES["neutral"]),
                "provider": "mock",
                "emotion": emotion,
                "learning_state": learning_state,
                "note": "Set LLM_PROVIDER + LLM_API_KEY in .env for real AI responses",
            }

        try:
            text = self._provider.chat(system, messages, max_tokens=800)
            return {
                "message": text,
                "provider": self._provider.name(),
                "emotion": emotion,
                "learning_state": learning_state,
            }
        except Exception as e:
            logger.error(f"AI provider error ({self._provider.name()}): {e}")
            return {
                "message": MOCK_RESPONSES.get(learning_state, MOCK_RESPONSES["neutral"]),
                "provider": "mock_fallback",
                "emotion": emotion,
                "learning_state": learning_state,
                "error": str(e),
            }

    def generate_question(
        self,
        subject: str,
        topic: str,
        difficulty: str,
        emotion_state: Dict,
    ) -> Dict:
        """Generate a question adapted to current emotion and difficulty."""

        learning_state = emotion_state.get("learning_state", "neutral")

        # Auto-adjust difficulty based on emotion
        if learning_state in ["frustrated", "anxious", "confused"] and difficulty != "easy":
            difficulty = "easy"
        elif learning_state in ["engaged", "curious"] and difficulty == "easy":
            difficulty = "medium"

        prompt = f"""Generate one {difficulty} question about {topic or subject}.
The student is currently {learning_state}.
Respond ONLY with valid JSON — no markdown, no backticks, no extra text:
{{
  "question": "...",
  "hint": "Think about...",
  "answer": "...",
  "explanation": "...",
  "topic": "{topic or subject}",
  "difficulty": "{difficulty}"
}}"""

        if not self._provider.is_ready():
            return self._mock_question(subject, topic, difficulty)

        try:
            raw = self._provider.chat("You are a question generator. Return only JSON.", 
                                       [{"role": "user", "content": prompt}], max_tokens=400)
            # Strip any accidental markdown fences
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            logger.error(f"Question generation error: {e}")
            return self._mock_question(subject, topic, difficulty)

    def get_hint(self, question: str, emotion_state: Dict, hint_level: int = 1) -> str:
        """Get a progressively more detailed hint."""
        learning_state = emotion_state.get("learning_state", "neutral")
        if learning_state in ["frustrated", "confused"]:
            hint_level = min(hint_level + 1, 3)

        defaults = {
            1: "Think about the key concept behind this question.",
            2: "Try breaking it into smaller parts. What's the very first step?",
            3: "Here's the approach: start by identifying what you know, then work towards the unknown.",
        }

        if not self._provider.is_ready():
            return defaults.get(hint_level, defaults[1])

        try:
            prompt = (f"Give a level-{hint_level} hint for: '{question}'\n"
                      f"Level 1=subtle nudge, 2=moderate help, 3=near-complete guidance.\n"
                      f"Student feels {learning_state}. "
                      f"{'Be very supportive and clear.' if learning_state in ['frustrated','confused'] else 'Keep it engaging.'}\n"
                      f"Reply with ONLY the hint text, nothing else.")
            return self._provider.chat("You are a helpful tutor giving hints.",
                                        [{"role": "user", "content": prompt}], max_tokens=150)
        except Exception as e:
            return defaults.get(hint_level, defaults[1])

    def _mock_question(self, subject: str, topic: str, difficulty: str) -> Dict:
        questions = {
            "math":    {"question": "If a train travels at 60 km/h for 2.5 hours, how far does it go?",
                        "hint": "Distance = Speed × Time", "answer": "150 km",
                        "explanation": "60 × 2.5 = 150 km"},
            "science": {"question": "What gas do plants absorb during photosynthesis?",
                        "hint": "Think about what plants take from the air",
                        "answer": "Carbon dioxide (CO₂)",
                        "explanation": "Plants convert CO₂ + water + sunlight → glucose + O₂"},
            "coding":  {"question": "What does a for loop do in Python?",
                        "hint": "Think about repetition",
                        "answer": "It repeats a block of code a set number of times",
                        "explanation": "for i in range(5): runs the block 5 times"},
        }
        q = questions.get(subject, {
            "question": f"Explain one key concept about {topic or subject} in your own words.",
            "hint": "Think about what you've learned so far",
            "answer": "Open-ended", "explanation": "Share your understanding!"
        })
        return {**q, "topic": topic or subject, "difficulty": difficulty}

    def is_ready(self) -> bool:
        return self._provider.is_ready()

    def provider_name(self) -> str:
        return self._provider.name()