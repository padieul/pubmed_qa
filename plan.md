## Main tasks:

### Data and Processing
- Extract all papers (all metadata) (Abdulghani)
- Create different datasets for experimentation
- standalone script that demonstrates storing data + embeddings in ES

### Architecture
- setting up containers: elasticsearch, kibana, middleware, frontend, (model - could be part of middleware)
- (research) retrieval mechanism: done by ES? custom retrieval? (Sushmitha)
- (research) how to connect LM with retrieval? Langchain? (Sushmitha)

### Model
- simple eval script - look at 3 different lms (Paul)
- a dataset of example questions, categorized by the 6 types from the project description
- (research) how to serve pytorch models? (Mahammad)
- simple demonstration of fine-tuning


### Evaluation Metrics (Sushmitha and Mahammad)
- BLEU (Sushmitha)
- ROUGE (Mahammad)
- BERTScore (Sushmitha)
- MoverScoere (Mahammad)