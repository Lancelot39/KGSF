# KGSF
KDD2020 Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion

# Environment
pytorch==1.3.0, torch_geometric==1.3.2

To be honest, most of errors derive from the wrong installation of the two packages

# Notation
The word embedding file **word2vec_redial.npy** can be produced by the following function dataset.prepare_word2vec(), or directly download from the netdisk https://drive.google.com/file/d/1BzwGgbUBilaEZXAu7e1SlvxSwcAfVe2w/view?usp=sharing

# Training
This model is trained by two steps, you should run the following code to pre-train the parameters by Mutual Information Maximization and then learn the recommendation task. Based on my experience, it will converge after 3 epochs pre-training and 3 epochs fine-tuning.

```python run.py```

Then you can run the following code to learn the conversation task, limitted by the small dataset, our model need about 30 epochs to covergence.

```python run.py --is_finetune True```

For convenience, our model will report the result on test data automatically after covergence.

# Thanks for your citation
https://arxiv.org/abs/2007.04032
