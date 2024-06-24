from encoders import BertTokenizer, BertModel

# Download and save tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('./bert-base-uncased-tokenizer')
model = BertModel.from_pretrained('bert-base-uncased')
model.save_pretrained('./bert-base-uncased-model')
from encoders import ElectraTokenizer, ElectraModel

# Download and save tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
tokenizer.save_pretrained('./electra-small-tokenizer')
model = ElectraModel.from_pretrained('google/electra-small-discriminator')
model.save_pretrained('./electra-small-model')
