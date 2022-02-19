import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
              
        # embedding feature vectors
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM takes embedded word vectors (of a specified size) as inputs 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # linear layer that maps the hidden state output dimension 
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
                
                   
    def forward(self, features, captions):
        
        # remove last token
        captions = captions[:, :-1]
        
        # embed captions
        embeds = self.word_embeddings(captions)
        
        # unsqueeze features and concatenate tensors
        embeds_features = torch.cat((features.unsqueeze(1), embeds), 1)
        
        # return output of LSTM
        lstm_out, _ = self.lstm(embeds_features)
        
        # return final caption of FC layer
        tag_outputs = self.hidden2tag(lstm_out)
        
        # return output
        return tag_outputs
    

    def sample(self, inputs, states=None, max_len=13):
        "Accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # list saving word ids indices
        tokens = []
        
        # predicted sentence for every tensor id
        for tensor_id in range(max_len):
            
            # return output of LSTM
            lstm_out, states = self.lstm(inputs, states)
            
            # return final caption of FC layer
            output_caption = self.hidden2tag(lstm_out)
            
            # squeeze output caption and returning index
            word_id_pred = output_caption.squeeze(1).argmax(dim=1)
            
            # add to tokens list
            tokens.append(word_id_pred.item())
            
            # update inputs
            inputs = self.word_embeddings(word_id_pred.unsqueeze(0))
            
        # return output
        return tokens