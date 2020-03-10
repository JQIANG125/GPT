from kashgari.embeddings import GPT2Embedding
from kashgari.corpus import CONLL2003ENCorpus
from kashgari.tasks.labeling.models import BiGRU_Model

train_x, train_y = CONLL2003ENCorpus.load_data('train')
valid_x, valid_y = CONLL2003ENCorpus.load_data('valid')

gpt2_embedding = GPT2Embedding('/content/model', sequence_length=128)
model = BiGRU_Model(gpt2_embedding)
model.fit(train_x, train_y, valid_x, valid_y, epochs=10)