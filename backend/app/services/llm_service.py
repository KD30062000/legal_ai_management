from typing import List, Dict, Any
import os
import asyncio
import google.generativeai as genai

class LLMService:
    def __init__(self):
        # Prefer existing env vars to avoid functional changes; support GEMINI_* as fallback
        self.model_name = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("GEMINI_MODEL")
            or "gemini-1.5-flash"
        )
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

        self.model = None
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)

    async def generate_response(self, prompt: str) -> str:
        """Generate a simple response from the LLM"""
        return await self._generate_text(prompt)

    async def generate_rag_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None,
    ) -> str:
        """Generate response using RAG with context documents and chat history"""

        # Build context from retrieved documents
        context = "\n\n".join(
            [
                f"Document: {doc['metadata'].get('filename', 'Unknown')}\nContent: {doc['content']}"
                for doc in context_documents
            ]
        )

        # Build chat history context
        history_context = ""
        if chat_history:
            history_context = "\n".join(
                [
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in chat_history[-5:]
                ]
            )

        # Create the RAG prompt
        system_prompt = (
            """You are a legal document AI assistant. You help users understand and analyze their company's legal documents. 

Guidelines:
1. Always base your answers on the provided document context
2. If information isn't available in the context, clearly state that
3. Provide specific references to documents when possible
4. Use clear, professional language suitable for business contexts
5. Consider the chat history to maintain conversation continuity
6. Focus on practical implications and actionable insights

Available Document Context:
{context}

Recent Chat History:
{history}"""
        )

        prompt = (
            system_prompt.format(context=context, history=history_context)
            + "\n\nUser Query:\n"
            + query
        )

        result = await self._generate_text(prompt)
        # Normalize and ensure we always return something readable
        if not result or not result.strip():
            return "I could not generate a response. Ensure documents exist for this company and the LLM API key is configured."
        # Preserve error message style from previous implementation
        return result if not result.startswith("Error generating response:") else (
            f"I apologize, but I encountered an error while processing your request: {result.replace('Error generating response: ', '')}"
        )

    async def generate_structured_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a structured summary of multiple documents"""
        
        context = "\n\n".join([
            f"Document: {doc['metadata'].get('filename', 'Unknown')}\nContent: {doc['content'][:2000]}..."
            for doc in documents
        ])

        prompt = f"""
        Analyze the following legal documents and provide a structured summary in the following format:

        1. DOCUMENT OVERVIEW
        - List of documents analyzed
        - Document types and purposes

        2. KEY PARTIES
        - Primary parties involved across documents
        - Their roles and relationships

        3. CRITICAL TERMS & CONDITIONS
        - Most important terms from all documents
        - Cross-references between documents

        4. IMPORTANT DATES & DEADLINES
        - All significant dates mentioned
        - Renewal dates, expiration dates, etc.

        5. RIGHTS & OBLIGATIONS
        - Key rights for each party
        - Main obligations and responsibilities

        6. RISK FACTORS & CONSIDERATIONS
        - Potential legal risks
        - Areas requiring attention

        7. RECOMMENDATIONS
        - Suggested actions
        - Areas for further review

        Documents:
        {context}
        """

        try:
            response = await self.generate_response(prompt)
            return {
                "summary": response,
                "document_count": len(documents),
                "generated_at": "now"
            }
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "document_count": len(documents),
                "generated_at": "now"
            }

    async def _generate_text(self, prompt: str) -> str:
        """Helper to call the Gemini model in a non-blocking way and return plain text."""
        if not self.model:
            return "Error generating response: Missing API key"

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            # Primary text field
            if hasattr(response, "text") and response.text:
                return response.text

            # Fallback: try to extract from candidates if needed
            candidates = getattr(response, "candidates", None)
            if candidates:
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    if not content:
                        continue
                    parts = getattr(content, "parts", None)
                    if not parts:
                        continue
                    texts = []
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            texts.append(part["text"])
                        elif hasattr(part, "text"):
                            texts.append(part.text)
                    if texts:
                        return "\n".join(texts)
            return ""
        except Exception as e:
            return f"Error generating response: {str(e)}"