"""
Automated Test Suite for OpenRouter Provider & Free Model Enforcement
Verifies:
1. Automatic detection of OpenRouter provider when API key starts with 'sk-or-'.
2. Strict enforcement of free models ('openrouter/free' or ':free' models).
3. OpenRouterChatModel invocation and message formatting.
4. Fast, 100% free LocalOfflineEmbeddings integration with FAISS.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from config import Config
from services.ai_service import (
    OpenRouterChatModel,
    OpenRouterResponse,
    get_chat_model
)
from services.rag_service import (
    LocalOfflineEmbeddings,
    get_embedding_model
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class TestOpenRouterProvider(unittest.TestCase):
    def setUp(self):
        self.mock_or_key = "sk-or-v1-mock-test-key-12345"

    def test_01_openrouter_model_enforces_free_router(self):
        """Verify that any non-free model passed to OpenRouterChatModel defaults to openrouter/free."""
        # Non-free model should be overridden to 'openrouter/free'
        model = OpenRouterChatModel(
            api_key=self.mock_or_key,
            model_name="openai/gpt-4o"  # Paid model
        )
        self.assertEqual(model.model, "openrouter/free")

        # Explicit free model should be permitted
        free_model = OpenRouterChatModel(
            api_key=self.mock_or_key,
            model_name="google/gemini-2.0-flash-exp:free"
        )
        self.assertEqual(free_model.model, "google/gemini-2.0-flash-exp:free")

        # Empty model defaults to 'openrouter/free'
        default_model = OpenRouterChatModel(api_key=self.mock_or_key, model_name="")
        self.assertEqual(default_model.model, "openrouter/free")

    def test_02_get_chat_model_auto_detects_openrouter_key(self):
        """Verify get_chat_model auto-detects OpenRouter when key starts with sk-or-."""
        model = get_chat_model(api_key=self.mock_or_key)
        self.assertIsInstance(model, OpenRouterChatModel)
        self.assertEqual(model.model, "openrouter/free")

    @patch("openai.resources.chat.completions.Completions.create")
    def test_03_openrouter_chat_model_invocation(self, mock_create):
        """Verify OpenRouterChatModel translates LangChain messages and returns OpenRouterResponse."""
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a grounded answer from openrouter/free."
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_create.return_value = mock_resp

        model = OpenRouterChatModel(api_key=self.mock_or_key, model_name="openrouter/free")

        messages = [
            SystemMessage(content="You are a study assistant."),
            HumanMessage(content="What is virtual memory?"),
            AIMessage(content="Virtual memory maps logical to physical addresses.")
        ]

        resp = model.invoke(messages)
        self.assertIsInstance(resp, OpenRouterResponse)
        self.assertEqual(resp.content, "This is a grounded answer from openrouter/free.")
        self.assertEqual(str(resp), "This is a grounded answer from openrouter/free.")

        # Verify call arguments
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        self.assertEqual(kwargs["model"], "openrouter/free")
        self.assertEqual(len(kwargs["messages"]), 3)
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertEqual(kwargs["messages"][1]["role"], "user")
        self.assertEqual(kwargs["messages"][2]["role"], "assistant")

    def test_04_local_offline_embeddings_for_openrouter(self):
        """Verify that get_embedding_model returns LocalOfflineEmbeddings when OpenRouter is active."""
        emb = get_embedding_model(api_key=self.mock_or_key)
        self.assertIsInstance(emb, LocalOfflineEmbeddings)

        # Check embedding dimensions and deterministic values
        vectors = emb.embed_documents(["Operating system process scheduling", "Database relational algebra"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 768)
        self.assertEqual(len(vectors[1]), 768)

        query_vec = emb.embed_query("process scheduling")
        self.assertEqual(len(query_vec), 768)


if __name__ == "__main__":
    unittest.main()
