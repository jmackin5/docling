# # Needed for older version of MacOS to avoid MPS backend usage in PyTorch
# import os
# # remove fallback variable and try to force CPU usage
# os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# # import torch and try to disable MPS backend flags before docling imports it
# import torch
# torch.backends.mps.is_available = lambda : False
# torch.backends.mps.is_built = lambda : False
# try:
#     torch.set_default_device("cpu")
# except Exception:
#     pass


from thrive_ai import ThriveAiDocling

docling = ThriveAiDocling()

processing_docs = True 

if processing_docs: 
    online_context = docling.process_pdf("https://arxiv.org/pdf/2408.09869")
    docling.save_markdown_to_file(online_context, "output.md")


question = "Is the model pipeline extensible ? "

online_context = docling.load_markdown_from_file("output.md")
answer,verification,gen = docling.answer_question(question, online_context)

print("GENERATION TYPE:", gen)
print("\nVERIFICATION SCORE:\n", verification['score'])
print("ANSWER:\n", answer)

