"""
inference.py — Load fine-tuned chatbot from Hugging Face & generate responses
==============================================================================
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import re


HF_REPO_ID  = "hari-krishna-ai/my-chatbot"   
FALLBACK     = "HF_TOKEN"  
MAX_LENGTH   = 200
MAX_NEW_TOKENS = 80
TEMPERATURE  = 0.8
TOP_P        = 0.92
TOP_K        = 50


class ChatBot:
    """
    Production-ready chatbot class.
    Maintains conversation history across turns for context-aware replies.
    """

    def __init__(self, repo_id: str = HF_REPO_ID, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history: List[str] = []       # Keeps last N exchanges
        self.max_history = 6               # Limit history to avoid OOM on CPU

        print(f"🤖 Loading chatbot from: {repo_id}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
            self.model     = AutoModelForCausalLM.from_pretrained(repo_id)
        except Exception as e:
            print(f"   ⚠️  Could not load '{repo_id}': {e}")
            print(f"   🔄 Falling back to: {FALLBACK}")
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK)
            self.model     = AutoModelForCausalLM.from_pretrained(FALLBACK)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        print(f"   ✅ Chatbot ready on {self.device.upper()}")

    def _build_prompt(self, user_input: str) -> str:
        """Build full prompt with conversation history for context."""
        context = ""
        for h in self.history[-self.max_history:]:
            context += h + " "
        prompt = f"{context}<|user|> {user_input} <|bot|>"
        return prompt

    def _clean_response(self, text: str) -> str:
        """Strip special tokens and return clean bot reply."""
        # Extract text after last <|bot|> tag
        if "<|bot|>" in text:
            text = text.split("<|bot|>")[-1]
        # Remove EOS / special tokens
        text = re.sub(r"<\|.*?\|>", "", text)
        text = text.replace(self.tokenizer.eos_token or "", "").strip()
        # Remove empty lines
        text = " ".join(text.split())
        return text if text else "I'm not sure how to respond to that. Can you rephrase?"

    @torch.no_grad()
    def chat(self, user_input: str) -> str:
        """
        Generate a response to user_input.
        Maintains conversation history automatically.
        """
        user_input = user_input.strip()
        if not user_input:
            return "Please say something!"

        prompt = self._build_prompt(user_input)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(self.device)

        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.3,     # Reduce repetitive responses
            no_repeat_ngram_size=3,
        )

        # Decode only newly generated tokens
        new_tokens  = output_ids[0][inputs["input_ids"].shape[-1]:]
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        response    = self._clean_response(full_output)

        # Update history
        self.history.append(f"<|user|> {user_input} <|bot|> {response}")

        return response

    def reset(self):
        """Clear conversation history to start a fresh session."""
        self.history = []

    def get_history(self) -> List[dict]:
        """Return conversation history as list of dicts for display."""
        parsed = []
        for turn in self.history:
            try:
                user_part = turn.split("<|bot|>")[0].replace("<|user|>", "").strip()
                bot_part  = turn.split("<|bot|>")[1].strip()
                parsed.append({"user": user_part, "bot": bot_part})
            except IndexError:
                continue
        return parsed


def generate_response(text: str, bot: Optional[ChatBot] = None) -> str:
    """
    Standalone function — creates a bot instance if not provided.
    Useful for one-off calls from app.py.
    """
    if bot is None:
        bot = ChatBot()
    return bot.chat(text)


# ── Quick CLI test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖 AI Chatbot — Type 'quit' to exit, 'reset' to clear history\n")
    bot = ChatBot()
    while True:
        user_in = input("You: ").strip()
        if user_in.lower() in ("quit", "exit", "q"):
            print("Bot: Goodbye! 👋")
            break
        elif user_in.lower() == "reset":
            bot.reset()
            print("Bot: History cleared. Starting fresh!")
        else:
            reply = bot.chat(user_in)
            print(f"Bot: {reply}\n")
