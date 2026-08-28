from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

class FinanceRAGRunner:
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def invoke(self, input_data: dict) -> dict:
        user_input = input_data.get("input", "")
        docs = self.retriever.invoke(user_input)
        context_text = "\n\n".join(doc.page_content for doc in docs)
        answer = self.chain.invoke({"context": context_text, "input": user_input})
        return {"answer": answer, "context": docs}

class FinanceRAGChain:
    def __init__(self, model_name: str = config.LLM_MODEL):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
                "Anda adalah Asisten Analis Keuangan Profesional yang sangat teliti.\n"
                "Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan konteks dokumen keuangan yang disediakan di bawah ini.\n\n"
                "Aturan Utama:\n"
                "1. Jawab HANYA berdasarkan informasi faktual yang ada dalam konteks.\n"
                "2. Jika jawaban tidak ada pada konteks, katakan dengan jujur: 'Informasi tersebut tidak ditemukan dalam dokumen yang diunggah.' Jangan mencoba mengarang data.\n"
                "3. Jika menyangkut angka atau transaksi, sebutkan angka secara presisi beserta mata uangnya.\n"
                "4. Sertakan referensi halaman dokumen asal jika tersedia pada metadata konteks.\n\n"
                "Konteks Dokumen Keuangan:\n"
                "{context}"
            ),
            ("human", "{input}")
        ])

    def create_chain(self, retriever):
        return FinanceRAGRunner(retriever, self.llm, self.prompt_template)