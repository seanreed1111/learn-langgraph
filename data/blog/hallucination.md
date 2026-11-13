---
title: Extrinsic Hallucinations in Language Models
---
# Extrinsic Hallucinations in Language Models

Hallucination in language models means generating content that is fabricated, unfaithful or not grounded in the provided context or in world knowledge. It is useful to separate two kinds. In-context hallucination is when the output contradicts the source content given in the prompt. Extrinsic hallucination is when the output is not grounded in the pre-training data, so the model must be factual about the world or admit that it does not know.

## Causes

Models hallucinate partly because pre-training data contains errors, and partly because fine-tuning on new knowledge that the model did not already hold can increase fabricated answers. Training objectives reward fluent completion, not calibrated uncertainty.

## Detection and mitigation

Retrieval augmented generation reduces hallucination by supplying evidence in the prompt. Other approaches include sampling several answers and checking consistency, asking the model to cite sources, verifying claims against a knowledge base, and training models to say they do not know when unsure. Checking whether an answer is grounded in the retrieved context is a common production safeguard.
