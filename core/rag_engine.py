from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from core.config import LLM_MODEL, TEMPERATURE


class RAGEngine:
    def __init__(self, retriever):
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=TEMPERATURE
        )

        self.retriever = retriever

        system_prompt = """
        تو یک دستیار هوشمند هستی.
        
        فقط از اطلاعات داده شده در context استفاده کن.
        اگر پاسخ در متن وجود نداشت بگو:
        "نمی‌دانم"
        
        context:
        {context}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            prompt
        )

        self.rag_chain = create_retrieval_chain(
            self.retriever,
            question_answer_chain
        )

    def ask_question(self, question: str):
        response = self.rag_chain.invoke(
            {"input": question}
        )

        answer = response["answer"]
        sources = response["context"]

        return {
            "answer": answer,
            "sources": sources
        }
