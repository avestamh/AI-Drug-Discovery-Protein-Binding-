'''Defines the DeepDTA model, consisting of CNN-based encoders for 
ligand and protein sequences, a multi-head attention mechanism,
 and a fully connected regression module for predicting binding affinity.'''

import torch.nn as nn
import torch


class mutil_head_attention(nn.Module):
    def __init__(self, head=16, conv=32):
        super(mutil_head_attention, self).__init__()
        self.conv = conv
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.d_a = nn.Linear(self.conv * 8, self.conv * 8 * head)
        self.p_a = nn.Linear(self.conv * 8, self.conv * 8 * head)
        self.scale = torch.sqrt(torch.FloatTensor([self.conv * 8])).cuda()

    def forward(self, drug, protein):

        # bsz = batch_size, d_ef = 96 (embedding feature), d_il = latent feature
        bsz, d_ef = drug.shape
        # bsz = batch_size, p_ef = 96 (embedding feature), p_il = latent feature
        bsz, p_ef = protein.shape

        drug_att = self.relu(self.d_a(drug)).view(bsz, d_ef, self.head)
        protein_att = self.relu(self.p_a(protein)).view(bsz, self.head, p_ef)

        interaction_map = self.tanh(torch.matmul(
            drug_att, protein_att) / self.scale)

        Compound_atte = self.tanh(torch.sum(interaction_map, 2))
        Protein_atte = self.tanh(torch.sum(interaction_map, 1))

        drug = drug * Compound_atte
        protein = protein * Protein_atte

        return drug, protein



# ------------------conv (used for encoding concat layer)-----------------------



class conv_layer(nn.Module):
    def __init__(self, input_filter, output_filter):
        super(conv_layer, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(input_filter, input_filter*2,
                      kernel_size=3, padding="valid", stride=1),
            nn.ReLU(),

        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(input_filter*2, input_filter*3,
                      kernel_size=5, padding="valid", stride=1),
            nn.ReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(input_filter*3, output_filter,
                      kernel_size=8, padding="valid", stride=1),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(6*input_filter, 1024)

    def forward(self, x):

        x = self.conv_layer1(x)
        # creating a skip connection for the first conv layer by taking the max value from each channel
        x1 = torch.max(x, 2).values
        x = self.conv_layer2(x)
        # creating a skip connection for the second conv layer by taking the max value from each channel
        x2 = torch.max(x, 2).values
        x = self.conv_layer3(x)
        # taking the max value from each channel of the third conv layer
        x3 = torch.max(x, 2).values
        # concatenating the data from all conv layers and preparing it for the final layer
        x = torch.cat((x1, x2, x3), 1)

        return self.final_layer(x)



#--------------------- Encoder (use to encode Drug and Protein) ----------------------

class Encoder(nn.Module):
    def __init__(self, size, max_len, num_filter, filter_length, embedding_feat):
        super(Encoder, self).__init__()
        self.embedding_feat = embedding_feat
        self.max_len = max_len
        self.embedding_layer = nn.Embedding(size+1, embedding_feat)
        self.embedding_layer.weight.data.uniform_(-1, 1)
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(embedding_feat, num_filter,
                      filter_length, padding="valid", stride=1),
            nn.ReLU(),

        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(num_filter, num_filter*2, filter_length,
                      padding="valid", stride=1),
            nn.ReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(num_filter*2, num_filter*3,
                      filter_length, padding="valid", stride=1),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(2*num_filter*3, 256)

    def forward(self, x):

        # creating the embedding vector based on the drug/target's character indexes
        x = self.embedding_layer(x)

        # reshaping the tensor to get it ready for 1d convolution
        x = x.view(-1, self.embedding_feat, self.max_len)
        x = self.conv_layer1(x)
        # taking the skip connection for conv layer 1 by taking the max value from each channel
        x1 = torch.max(x, 2).values
        x = self.conv_layer2(x)
        # taking the skip connection for conv layer 2 by taking the max value from each channel
        x2 = torch.max(x, 2).values
        x = self.conv_layer3(x)
        # for conv layer 3 by taking the max value from each channel
        x3 = torch.max(x, 2).values
        # concatenating all the feature in the axis=1 or dimention=1
        x_feat = torch.cat((x1, x2, x3), 1)

        return self.final_layer(x_feat), x



#---------------------------- Deep DTA Model----------------------------------------


# dim=-1 is the right most dimension

class DeepDTA(nn.Module):
    def __init__(
        self, smile_size=64, seq_size=25, max_smile_len=100, max_seq_len=1200, num_filter=32, kernel_smile=8, kernel_seq=8, embedding_feat=128
    ):
        super(DeepDTA, self).__init__()
        self.smile_encode = Encoder(
            smile_size, max_smile_len, num_filter, kernel_smile, embedding_feat)
        self.protein_encode = Encoder(
            seq_size, max_seq_len, num_filter, kernel_seq, embedding_feat)
        self.concat_encode = conv_layer(num_filter*3, num_filter*3)
        self.attention = mutil_head_attention(head=16, conv=num_filter)
        self.final_mod = nn.Sequential(
            nn.Linear(16*num_filter*3, 2048),
            # nn.Linear(38*num_filter , 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
        )
        self.affinity = nn.Linear(512, 1)
        # torch.nn.init.normal_(self.affinity.weight)

    def forward(self, xD, xP):
        # getting drug encoding for final layer and drug features concat module
        xD, x1 = self.smile_encode(xD)

        # getting target encoding for final layer and target features concat module
        xP, x2 = self.protein_encode(xP)

        # concatenating the drug and target features for concat module
        x_con = torch.cat((x1, x2), -1)

        # getting the concatenated module encoding for the final layer
        x_con = self.concat_encode(x_con)

        xD, xP = self.attention(xD, xP)

        x = torch.cat((xD, x_con, xP), 1)

        # x = torch.cat((xD, x_con, xP),1) # concatenating all the encoding for the final layer

        # propagating through the final module (fully connected layer)
        x1 = self.final_mod(x)

        x = self.affinity(x1)  # predicting the affinity value
        return x
