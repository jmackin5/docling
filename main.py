from thrive_ai import ThriveAiDocling

docling = ThriveAiDocling()

processing_docs = False 

if processing_docs: 
    online_context = docling.process_pdf("https://arxiv.org/pdf/2408.09869")
    docling.save_markdown_to_file(online_context, "output.md")


question = "Is the model pipeline extensible ? "

online_context = docling.load_markdown_from_file("output.md")
answer,verification,gen = docling.answer_question(question, online_context)

print("GENERATION TYPE:", gen)
print("\nVERIFICATION SCORE:\n", verification['score'])
print("ANSWER:\n", answer)


# answer = docling.ask_document_question(question, online_context)
# verification_result = docling.verification_prompt(question, answer, online_context)
# print(answer)
# print("\nVERIFICATION RESULT:\n", verification_result)