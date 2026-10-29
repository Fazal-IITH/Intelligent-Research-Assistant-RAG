import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

def generate_answer(retrieved_chunks, question):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
                You are a helpful AI assistant.

        The following context comes from an uploaded PDF.

        Answer the user's question using ONLY the provided context.

        If the context partially answers the question, provide the best answer you can and clearly state that it is based on the available context.

        If the context allows a reasonable inference, you may make that inference, but do not invent new facts.

        Only reply:
        "I couldn't find that information in the uploaded document."
        when the context contains no relevant information at all.

        Context:
        {context}

        Question:
        {question}
                """
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents= prompt
    )
    return response.text

