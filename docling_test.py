
#pip3 install docling
#sudo apt-get install libgl1-mesa-glx libglib2.0-0
#
#Free AI
#curl -fsSL https://ollama.ai/install.sh | sh
#ollama pull llama3.2:3b
from docling.document_converter import DocumentConverter
import openai
import json 
import requests

# #Need OpenAi key
# def ask_document_question(question, markdown_content):
#     prompt = f"""
#     Based on this employee handbook content:
    
#     {markdown_content}
    
#     Question: {question}
    
#     Please provide a specific answer based only on the information in the document.
#     """
#     from openai import OpenAI

#     client = OpenAI(
#         # This is the default and can be omitted
#         api_key=os.environ.get("OPENAI_API_KEY"),
#     )

#     # Use your preferred LLM API
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     # # Use your preferred LLM API
#     # response = openai.chat.completions.create(
#     #     model="gpt-4",
#     #     messages=[{"role": "user", "content": prompt}]
#     # )
#     return response.choices[0].message.content


def ask_document_question_free(question, markdown_content):
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


def process_pdf(source):
    #source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
    converter = DocumentConverter()
    result = converter.convert(source)

    return result.document.export_to_markdown()
    #print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

def save_markdown_to_file(content, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_file

def load_markdown_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# online_context = process_online_pdf("https://arxiv.org/pdf/2408.09869")
# answer = ask_document_question_free("Can you summarize the processing pipeline ? ", online_context)
# print(answer)


online_context = process_pdf("https://arxiv.org/pdf/2408.09869")


save_markdown_to_file(online_context, "output.md")
online_context = load_markdown_from_file("output.md")
answer = ask_document_question_free("How much PTO do i have in my first year ? ", online_context)
print(answer)