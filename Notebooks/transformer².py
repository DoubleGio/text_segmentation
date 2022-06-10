"""
[pdf](https://arxiv.org/pdf/2110.07160.pdf)
Transformer² model (without S_single & L_topic):
    1. Obtain the sentence embeddings (using pretrained BERT).
        >>> sentence1 = 'sentence one'; sentence2 = 'yet another one'
        1.1. Pairwise tokenize (skip last sentence I guess?): 
            >>> tokens = tokenizer(text=sentence1, textpair=sentence2, padding=True, return_tensors='pt')
            >>> ['[CLS]', 'sentence', 'one', '[SEP]', 'yet', 'another', 'one', '[SEP]']
        1.2. Get sentence embedding from CLS token:
            >>> out = model(**tokens)
            >>> out.last_hidden_state[:, 0, :] # (sentences, tokens, hidden_size)
    2. Train a transformer model to classify whether each sentence is a segment boundary.
        2.1. Create a transformer model:
            >>> y_seg = Sigmoid(Linear2(TransformerΘ(S)))
            >>> Θ = {'nhead': 24, 'num_encoder_layers': 5, 'dim_feedforward': 1024}
        2.2. Create a loss function:
            >>> loss_fn = binary cross entropy loss
        'For the segmentation predictions, 70% of the inner sentences were randomly masked,
        while all the begin sentences were not masked in order to address the imbalance class problem.'
        basically, remove 70% of the non-boundary sentences for training.
"""
