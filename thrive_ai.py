
#pip3 install docling
#sudo apt-get install libgl1-mesa-glx libglib2.0-0
#
#Free AI
#curl -fsSL https://ollama.ai/install.sh | sh
#ollama pull llama3.2:3b
# ...existing code...
import os
# remove fallback variable and try to force CPU usage
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# import torch and try to disable MPS backend flags before docling imports it
import torch
torch.backends.mps.is_available = lambda : False
torch.backends.mps.is_built = lambda : False
try:
    torch.set_default_device("cpu")
except Exception:
    pass


class ThriveAiDocling():
    def __init__(self):
        self.memory_file = "memory.json"
        self.verified_memory = []
        pass



    def load_memory(self):
        import json 
        import os

        if not os.path.exists(self.memory_file):
            return {}
        with open(self.memory_file, "r") as f:
            return json.load(f)

    def save_memory(self,memory_data):
        import json
        with open(self.memory_file, "w") as f:
            json.dump(memory_data, f, indent=4)

    def add_verified_item(self,question, answer, score):
        memory = self.load_memory()
        memory[question] = {
            "answer": answer,
            "score": score
        }
        self.save_memory(memory)


    # #Need OpenAi key
    def ask_document_question(self, question, markdown_content):
        prompt = f"""
        Based on this document content:
        
        {markdown_content}
        
        Question: {question}
        
        Please provide a specific answer based only on the information in the document.
        """
        from openai import OpenAI
        import os

        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Use your preferred LLM API
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content


    def ask_document_question_free(self,question, markdown_content):
        import json 
        import requests 

        prompt = f"""Based on this document:

    {markdown_content[:3000]}...

    Question: {question}

    Answer based only on the document above:"""
        
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2:3b',
                'prompt': prompt,
                'stream': False
            })
        
        return json.loads(response.text)['response']

    def process_pdf(self, source):
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()

        # If a list/tuple of sources is passed, convert each and combine results
        if isinstance(source, (list, tuple)):
            print(f"Processing {len(source)} documents...")
            parts = []
            for idx, src in enumerate(source, start=1):
                try:
                    result = converter.convert(src)
                    md = result.document.export_to_markdown()
                    # Add a simple separator between documents
                    parts.append(f"<!-- Document {idx}: {src} -->\n\n" + md)
                except Exception as e:
                    # keep behavior simple: warn and continue with other docs
                    print(f"Warning: failed to convert {src}: {e}")
            return "\n\n---\n\n".join(parts)

        else:
            print(f"Processing single document...")
            # Single source behavior (unchanged)
            result = converter.convert(source)
            return result.document.export_to_markdown()

    def save_markdown_to_file(self, content, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_file

    def load_markdown_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


    def chunk_text(self, text, max_chars=1500):
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


    def verification_prompt(self, question, answer, document_text):
        import openai
        import json 
        
        chunks = self.chunk_text(text = document_text, max_chars=1500)

        verification_prompt = f"""

        You are a Verification Agent.

        Your job is to check whether an answer is factually supported by the provided document chunks.

        Return your answer **as strict JSON** with the following keys:
        - score: integer (0–100)
        - supported_claims: dictionary following this format 'pdf_sections': [list of section titles where supported chunk is found], 'chunks': [list of supporting chunks]
        - unsupported_claims: dictionary following this format  'pdf_sections': [list of section titles where unsupported chunk is found], 'chunks': [list of unsupported chunks]
        - contradictory_claims: dictionary following this format  'pdf_sections': [list of section titles where contradictory chunk is found], 'chunks': [list of contradictory chunks]
        - verdict: "SUPPORTED" | "PARTIALLY SUPPORTED" | "UNSUPPORTED"

        Scoring rubric:
        100 = fully supported
        70–99 = mostly supported
        40–69 = mixed
        1–39 = mostly unsupported
        0 = fabricated
    

        Document Chunks:
        {chunks}

        Question: {question}
        Answer:
        {answer}

        Now evaluate and output ONLY valid JSON.
        """

        verification = openai.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": verification_prompt}]
        )

        verification_result = verification.choices[0].message.content

        data = json.loads(verification_result)

        return data
    


    def embed(self, text):

        from openai import OpenAI
        client = OpenAI()

        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding




    def cosine(self,a, b):
        import numpy as np

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    #Need to adjust threshold to determine memory match sensitivity
    def search_memory(self,query_embedding, memory, threshold=0.85):
        best = None
        best_score = 0

        for item in memory:
            sim = self.cosine(query_embedding, self.embed(item) ) #item["embedding"])
            if sim > best_score:
                best_score = sim
                best = item

        if best and best_score >= threshold:
            return memory[best]
        return None
    
    def answer_question(self,question,markdown_content):
        q_embed = self.embed(question)

        # # 1. Check if similar verified Q exists
        match = self.search_memory(q_embed, self.load_memory())
        if match:
            return match["answer"],match, "FROM_MEMORY" #match["answer"], "FROM_MEMORY"

        # 2. Else run normal RAG (Docling + LLM)
        llm_answer = self.ask_document_question(question,markdown_content)  # Responder agent
        verification = self.verification_prompt(question, llm_answer, markdown_content)  # JSON dict

        score = verification["score"]

        # 3. If answer is high-confidence, store it
        if score >= 85:
            self.add_verified_item(question, llm_answer, score)

        return llm_answer,verification,"NEW_GENERATION"
