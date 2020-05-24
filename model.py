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
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
    
    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        lstm_out, self.hidden = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        # initialize the output
        output = []
        # batch_size
        batch_size = inputs.shape[0]
        
        # initialize hidden state
        self.hidden = torch.zeros((1, batch_size, self.hidden_dim)).to(device)
        self.cell_state = torch.zeros((1, batch_size, self.hidden_dim)).to(device)
    
        while True: 
            # pass through lstm unit(s)
            lstm_out, (self.hidden, self.cell_state) = self.lstm(inputs, (self.hidden, self.cell_state))
            
            # pass through linear unit
            outputs = self.linear(lstm_out)
            
            # predict the most likely next word
            outputs = outputs.squeeze(1)
            _, max_indice = torch.max(outputs, dim=1) 
            
            # storing the word predicted
            output.append(max_indice.cpu().numpy()[0].item()) 
            
            if (max_indice == 1 or len(output) >= max_len):
                # if reached the max length or predicted the end token
                break
            
            ## embed the last predicted word to be the new input of the lstm
            inputs = self.embed(max_indice) 
            inputs = inputs.unsqueeze(1)
            
        return output