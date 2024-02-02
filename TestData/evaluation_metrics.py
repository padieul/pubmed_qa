import evaluate
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
#Sample Reference and Candidate 
references = [["Hello, how are you?", "This is a test sentence.", "BERTScore is awesome!"]]
candidates = [["Hello, how are you?", "This is a test sentence.", "BERTScore is amazing!"]]

# Add batch to BLEU metric
bleu.add_batch(predictions=candidates, references=references)

# Compute BLEU score
bleu_score = bleu.compute()
print("BLEU Score:", bleu_score)

# Add batch to BERTScore metric
bertscore.add_batch(predictions=candidates, references=references)

# Compute BERTScore
bertscore_results = bertscore.compute()
print("BERTScore Results:", bertscore_results)