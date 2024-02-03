import evaluate

#Sample predictions and references
predictions = ["Hello, how are you?", "This is a test sentence.", "BERTScore is awesome!"]
references = ["Hello, how are you?", "This is a test sentence.", "BERTScore is amazing!"]
references_bleu = [["Hello, how are you?"], ["This is a test sentence."], ["BERTScore is amazing!"]]


bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")


bleu_results = bleu.compute(predictions=predictions, references=references_bleu)
print("BLEU Score:", bleu_results)

bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
print("BERTScore Results:", bertscore_results)

rouge_results = rouge.compute(predictions=predictions, references=references)
print("ROUGE Results:", rouge_results)