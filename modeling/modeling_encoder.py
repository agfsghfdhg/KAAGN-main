import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spacy
import pickle
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
except:
    pass
from transformers import AutoModel, AutoTokenizer
from utils.layers import *
from utils.data_utils import get_gpt_token_num

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}
try:
    MODEL_CLASS_TO_NAME['albert'] =  list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
except:
    pass

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}


class LSTMTextEncoder(nn.Module):
    pool_layer_classes = {'mean': MeanPoolLayer, 'max': MaxPoolLayer}

    def __init__(self, vocab_size=1, emb_size=300, hidden_size=300, output_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, pretrained_emb_or_path=None, freeze_emb=True,
                 pool_function='max', output_hidden_states=False):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_hidden_states = output_hidden_states
        assert not bidirectional or hidden_size % 2 == 0

        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                pretrained_emb_or_path = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb = nn.Embedding.from_pretrained(pretrained_emb_or_path, freeze=freeze_emb)
            emb_size = emb.weight.size(1)
        else:
            emb = nn.Embedding(vocab_size, emb_size)
        self.emb = EmbeddingDropout(emb, emb_p)
        self.rnns = nn.ModuleList([nn.LSTM(emb_size if l == 0 else hidden_size,
                                           (hidden_size if l != num_layers else output_size) // (2 if bidirectional else 1),
                                           1, bidirectional=bidirectional, batch_first=True) for l in range(num_layers)])
        self.pooler = self.pool_layer_classes[pool_function]()

        self.input_dropout = nn.Dropout(input_p)
        self.hidden_dropout = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        assert (lengths > 0).all()
        batch_size, seq_len = inputs.size()
        hidden_states = self.input_dropout(self.emb(inputs))
        all_hidden_states = [hidden_states]
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropout)):
            hidden_states = pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
            hidden_states, _ = rnn(hidden_states)
            hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True, total_length=seq_len)
            all_hidden_states.append(hidden_states)
            if l != self.num_layers - 1:
                hidden_states = hid_dp(hidden_states)
        pooled = self.pooler(all_hidden_states[-1], lengths)
        assert len(all_hidden_states) == self.num_layers + 1
        outputs = (all_hidden_states[-1], pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs

class Dependency_tree(nn.Module):

    def forward(self, text):
        nlp = spacy.load('en_core_web_sm')
        # https://spacy.io/docs/usage/processing-text
        document = nlp(text)
        seq_len = len(text.split())
        matrix = np.zeros((seq_len, seq_len)).astype('float32')

        for token in document:
            if token.i < seq_len:
                matrix[token.i][token.i] = 1
                # https://spacy.io/docs/api/token
                for child in token.children:
                    if child.i < seq_len:
                        matrix[token.i][child.i] = 1

        return matrix

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input, adj):
        support = self.linear(input)
        output = torch.matmul(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, input, adj):
        x = F.relu(self.gc1(input, adj))
        output = self.gc2(x, adj)
        return output

class QuestionEncoder(nn.Module):
    def __init__(self):
        # 加载RoBERTa模型和tokenizer
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("/home/xyNLP/data/kl/DRGN-main/modeling/aristo-roberta")
        self.model = AutoModel.from_pretrained("/home/xyNLP/data/kl/DRGN-main/modeling/aristo-roberta",output_hidden_states=True)


    def forward(self, question):

        # 使用tokenizer对句子进行编码和标记化
        tokens = self.tokenizer.encode(question, add_special_tokens=True)
        input_ids = torch.tensor(tokens).unsqueeze(0).cuda()  # 添加批次维度

        # 将输入传递给RoBERTa模型
        outputs = self.model(input_ids)

        # 获取句子的特征张量
        sentence_features = outputs.last_hidden_state.squeeze(0)

        # 仅保留单词数量的部分
        word_count = len(tokens) - 3  # 减去开头和结尾的特殊标记的数量
        sentence_features = sentence_features[:word_count, :]

        return sentence_features


class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in ('bert', 'roberta', 'albert')

        if self.model_type in ('lstm',):
            self.module = LSTMTextEncoder(**kwargs, output_hidden_states=True)
            self.sent_dim = self.module.output_size
        else:
            # self.module = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            ### chen begin
            # self.module = AutoModel.from_pretrained("danlou/aristo-roberta-finetuned-csqa", output_hidden_states=True)
            # self.module = AutoModel.from_pretrained("/home/xyNLP/data/kl/DRGN-main/modeling/aristo-roberta", output_hidden_states=True)
            ### chen end

            if model_name in ('roberta-large'):
                self.tokenizer = AutoTokenizer.from_pretrained("/home/xyNLP/data/kl/DRGN-main/modeling/aristo-roberta")
                self.module = AutoModel.from_pretrained("/home/xyNLP/data/kl/DRGN-main/modeling/roberta-large",
                                                        output_hidden_states=True)
            if model_name in ('aristo-roberta'):
                self.tokenizer = AutoTokenizer.from_pretrained("/home/xyNLP/data/kl/DRGN-main/modeling/aristo-roberta")
                self.module = AutoModel.from_pretrained("/home/xyNLP/data/kl/DRGN-main/modeling/aristo-roberta",
                                                        output_hidden_states=True)
            if from_checkpoint is not None:
                self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
            if self.model_type in ('gpt',):
                self.module.resize_token_embeddings(get_gpt_token_num())
            self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        layer_id: only works for non-LSTM encoders
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''

        if self.model_type in ('lstm',):  # lstm
            input_ids, lengths = inputs
            outputs = self.module(input_ids, lengths)
        elif self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.module(input_ids)
        else:  # bert / xlnet / roberta
            input_ids, attention_mask, token_type_ids, output_mask = inputs
            outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        if self.model_type in ('lstm',):
            sent_vecs = outputs[1]
        elif self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        elif self.model_type in ('albert',):
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = hidden_states[:, 0]
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = self.module.pooler(hidden_states)
        return sent_vecs, all_hidden_states

    def question_encoder(self, question):
        # 使用tokenizer对句子进行编码和标记化
        tokens = self.tokenizer.encode(question, add_special_tokens=True)
        input_ids = torch.tensor(tokens).unsqueeze(0).cuda()  # 添加批次维度

        # 将输入传递给RoBERTa模型
        outputs = self.module(input_ids)

        # 获取句子的特征张量
        sentence_features = outputs.last_hidden_state.squeeze(0)

        # 仅保留单词数量的部分
        word_count = len(tokens) - 3  # 减去开头和结尾的特殊标记的数量
        sentence_features = sentence_features[:word_count, :]

        return sentence_features

    def dependency_tree(self, text):
        nlp = spacy.load('en_core_web_sm')
        # https://spacy.io/docs/usage/processing-text
        document = nlp(text)
        seq_len = len(text.split())
        matrix = np.zeros((seq_len, seq_len)).astype('float32')

        for token in document:
            if token.i < seq_len:
                matrix[token.i][token.i] = 1
                # https://spacy.io/docs/api/token
                for child in token.children:
                    if child.i < seq_len:
                        matrix[token.i][child.i] = 1

        return matrix



def run_test():
    encoder = TextEncoder('lstm', vocab_size=100, emb_size=100, hidden_size=200, num_layers=4)
    input_ids = torch.randint(0, 100, (30, 70))
    lenghts = torch.randint(1, 70, (30,))
    outputs = encoder(input_ids, lenghts)
    assert outputs[0].size() == (30, 200)
    assert len(outputs[1]) == 4 + 1
    assert all([x.size() == (30, 70, 100 if l == 0 else 200) for l, x in enumerate(outputs[1])])
    print('all tests are passed')
