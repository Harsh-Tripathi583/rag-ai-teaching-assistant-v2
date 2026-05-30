import requests


class LLM:

    def __init__(self, model_name="llama3.2"):

        self.model_name = model_name

    def generate_response(self, query, context):

        #context = "\n\n".join(retrieved_chunks)
        
        prompt = f"""
You are an AI Teaching Assistant.

Use the provided context to answer the user's question.

The context may contain indirect, partial, or distributed
information relevant to the answer. Carefully analyze and
connect related pieces of information before answering.

Provide a clear and educational explanation.

Do not invent facts outside the provided context.

If the answer genuinely cannot be determined from the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        response = requests.post(

            "http://localhost:11434/api/generate",

            json={

                "model": self.model_name,

                "prompt": prompt,

                "stream": False
            }
        )

        result = response.json()

        return result["response"]