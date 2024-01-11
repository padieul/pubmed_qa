import openai
api_key = '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'
openai.api_key = api_key

def generate_answer(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-1106",  
        prompt=prompt,
        max_tokens=150  
    )
    return response.choices[0].text.strip()

# Example prompt
abstract= input("Enter an abstract")
prompt_example = "Analyse the following abstract" + abstract + "and generate 6 quetsions and short answers in the following format :1. Confirmation Questions [yes or no]: Yes/No questions require an understanding of a given context and deciding a boolean value for an answer, e.g., ‘Is Paris the capital of France?’, 2. Factoid-type Questions [what, which, when, who, how]: These usually begin with a ‘wh’-word. An answer then is commonly short and formulated as a single sentence. In some cases, returning a snippet of a document’s text already answers the question, e.g., ‘What is the capital of France?’, where a sentence from Wikipedia answers the question.3. List-type Questions: The answer is a list of items, e.g.,’Which cities have served as the capital of France throughout its history?’. Answering such questions rarely requires any answering generation if the exact list is stored in some document in the corpus already.4. Causal Questions [why or how]: Causal questions seek reasons, explanations, and elaborations on particular objects or events, e.g., ‘Why did Paris become the capital of France?’ Causal questions have descriptive answers that can range from a fewsentences to whole paragraphs.5. Hypothetical Questions: These questions describe a hypothetical scenario and usually start with ‘what would happen if’, e.g., ‘What would happen if Paris airport closes for a day?’. The reliability and accuracy of answers to these questions are typically low in most application settings. 6.Complex Questions: Sometimes a question requires multi-part reasoning by understanding the semantics of multiple text snippets to generate a correct answer, e.g., ‘What cultural and historical factors contributed to the development of the Louvre Museum as a world-renowned art institution?’, which requires inferring information from multiple documents to generate an answer'"

# Generate answer
answer = generate_answer(prompt_example)

# Display the result
print(f"Prompt: {prompt_example}")
print(f"Answer: {answer}")
