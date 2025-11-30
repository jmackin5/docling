
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


def process_pdf_basic(pdf_path):
    # Use default converter with minimal configuration
    converter = DocumentConverter()
    
    print(f"Processing {pdf_path}...")
    result = converter.convert(pdf_path)
    
    # Export to markdown
    markdown_content = result.document.export_to_markdown()
    
    return markdown_content, result

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

###If need to read and convert doc first 
# Process your PDF
# pdf_path = "Employee_Handbook_Fall_2025.pdf"

# try:
#     markdown_content, result = process_pdf_basic(pdf_path)
    
#     # Save the markdown output
#     with open("handbook_processed.md", "w", encoding="utf-8") as f:
#         f.write(markdown_content)
    
#     print("PDF processed successfully!")
#     print(f"Markdown saved to: handbook_processed.md")
#     print(f"Document has {len(result.document.pages)} pages")
    
# except Exception as e:
#     print(f"Error processing PDF: {e}")
#     print("Let's try a different approach...")


# # Load your processed markdown
# with open("handbook_processed.md", "r", encoding="utf-8") as f:
#     content = f.read()

# # Ask questions for free
# answer = ask_document_question_free("What can you tell me about this document? ", content)
# print(answer)

def process_online_pdf(source):
    #source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
    converter = DocumentConverter()
    result = converter.convert(source)

    return result.document.export_to_markdown()
    print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"


# online_context = process_online_pdf("https://arxiv.org/pdf/2408.09869")
# answer = ask_document_question_free("Can you summarize the processing pipeline ? ", online_context)
# print(answer)


online_context = process_online_pdf("Employee_Handbook_Fall_2025.pdf")
answer = ask_document_question_free("How much PTO do i have in my first year ? ", online_context)
print(answer)