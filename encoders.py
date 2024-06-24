from transformers import ElectraTokenizer, ElectraModel
from transformers import BertTokenizer, BertModel

def load_bert():
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased-tokenizer')
    model = BertModel.from_pretrained('./bert-base-uncased-model')
    return tokenizer, model

def load_electra():
    tokenizer = ElectraTokenizer.from_pretrained('./electra-small-tokenizer')
    model = ElectraModel.from_pretrained('./electra-small-model')
    return tokenizer, model

def embed_text_with_bert(text):
    bert_tokenizer, bert_model = load_bert()
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(inputs.shape)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()

def embed_text_with_electra(text):
    electra_tokenizer, electra_model = load_electra()
    inputs = electra_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(inputs.shape) 
    outputs = electra_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()