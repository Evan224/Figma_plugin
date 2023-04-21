import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x_dict = self.conv2(x, edge_index)
        x_dict = {key: global_mean_pool(x.x, x.batch) for key, x in x_dict.items()}
        x = torch.stack(x_dict.values(), dim=0).sum(dim=0)
        
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(input_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):

        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        return x


class Classification(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(54321)
        self.lin1 = Linear(input_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):

        x = self.lin1(x)
        x = self.activation(x)

        x = self.lin2(x)
        x = self.activation(x)

        x = self.lin3(x)

        return x


class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(54321)
        self.pos_embedding = Linear(2, 256)

    def forward(self, x):
        x = self.pos_embedding(x)
        # x = torch.flatten(self.pos_embedding(x.to(device))).unsqueeze(0)
        return x

class SizeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(54321)
        self.size_embedding = Linear(2, 256)

    def forward(self, x):
        x = self.size_embedding(x)
        return x


class TypeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(54321)
        self.type_embedding = nn.Embedding(23, 512)

    def forward(self, x):
        x = self.type_embedding(x.type(torch.IntTensor).to(device))
        return x


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # ('element', 'occupies', 'cell'): SAGEConv(-1, hidden_channels),
                ('element', 'aligns', 'alignment'): SAGEConv((-1, -1), hidden_channels),
                ('element', 'has', 'size'): SAGEConv((-1, -1), hidden_channels),
                ('element', 'with', 'element_grouping'): SAGEConv((-1, -1), hidden_channels),
                ('element', 'hwith', 'horizontal_grouping'): SAGEConv((-1, -1), hidden_channels),
                ('element', 'vwith', 'vertical_grouping'): SAGEConv((-1, -1), hidden_channels),
                ('element', 'belongs', 'multimodal_grouping'): SAGEConv((-1, -1), hidden_channels),
                ('alignment', 'rev_aligns', 'element'): SAGEConv((-1, -1), hidden_channels),
                ('size', 'rev_has', 'element'): SAGEConv((-1, -1), hidden_channels),
                ('element_grouping', 'rev_with', 'element'): SAGEConv((-1, -1), hidden_channels),
                ('horizontal_grouping', 'rev_hwith', 'element'): SAGEConv((-1, -1), hidden_channels),
                ('vertical_grouping', 'rev_vwith', 'element'): SAGEConv((-1, -1), hidden_channels),
                ('multimodal_grouping', 'rev_belongs', 'element'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(out_channels, out_channels)
        self.lin2 = Linear(hidden_channels, out_channels)



    def forward(self, x_dict, edge_index_dict, data, batch_size):

        for key in edge_index_dict.keys():
            edge_index_dict[key] = edge_index_dict[key].type(torch.LongTensor).to(device)

        res = 0
        x_dict_res = x_dict
        # print(edge_index_dict)
        edge_index_dict_copy = edge_index_dict.copy()
        for key in edge_index_dict_copy.keys():
            if edge_index_dict[key].shape[1] == 0:
                del edge_index_dict[key]
            # print(edge_index_dict[key])
        # print('=======', x_dict)
        for conv in self.convs:

            x_dict = conv(x_dict, edge_index_dict)
            # print('=======', x_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            if res == 0:
                x_dict_res = x_dict
            if res < 1:
                res += 1
            else:
                res = 0
                x_dict = {key: x + x_dict_res[key] for key, x in x_dict.items()}
                x_dict_res = x_dict  

        x_dict = {key: self.lin2(x) for key, x in x_dict.items()}


        if x_dict['element'] != []:
            #print(data['element'].batch, x_dict['element'].shape[0])
            #print(torch.LongTensor([0] * x_dict['element'].shape[0]), data['element'].batch)
            #exit()
            x_element = global_mean_pool(x_dict['element'], torch.LongTensor([0] * x_dict['element'].shape[0])) 
        else:
            print('ERROR: No element!')
        #     x_element = torch.zeros(1, 2536).to(device)

        if 'alignment' in x_dict.keys() and x_dict['alignment'].shape[0] != 0:

            x_dict_alignment = x_dict['alignment']
            alignment_batch = torch.LongTensor([0] * x_dict['alignment'].shape[0])

            # if the constraint type does not exist in the last graph, 
            # then we need to give a hint to let it know that what is 
            # the last index in the batch
            if alignment_batch[-1] != batch_size - 1:
                x_dict_alignment = torch.cat((x_dict_alignment, 
                                torch.zeros(1, x_element.shape[1]).to(device)), axis=0)
                alignment_batch = torch.cat((alignment_batch, 
                                torch.LongTensor([batch_size-1]).to(device)), axis=0)

            x_alignment = global_mean_pool(x_dict_alignment, alignment_batch)   
        else:
            x_alignment = torch.zeros_like(x_element).to(device)

        if 'size' in x_dict.keys() and x_dict['size'].shape[0] != 0:

            x_dict_size = x_dict['size']
            size_batch = torch.LongTensor([0] * x_dict['size'].shape[0])

            # if the constraint type does not exist in the last graph, 
            # then we need to give a hint to let it know that what is 
            # the last index in the batch
            if size_batch[-1] != batch_size - 1:
                x_dict_size = torch.cat((x_dict_size, 
                                torch.zeros(1, x_element.shape[1]).to(device)), axis=0)
                size_batch = torch.cat((size_batch, 
                                torch.LongTensor([batch_size-1]).to(device)), axis=0)


            x_size = global_mean_pool(x_dict_size, size_batch)   
        else:
            x_size = torch.zeros_like(x_element).to(device)

        if 'element_grouping' in x_dict.keys() and x_dict['element_grouping'].shape[0] != 0:
            x_dict_element_grouping = x_dict['element_grouping']
            element_grouping_batch = torch.LongTensor([0] * x_dict['element_grouping'].shape[0])

            # if the constraint type does not exist in the last graph, 
            # then we need to give a hint to let it know that what is 
            # the last index in the batch
            if element_grouping_batch[-1] != batch_size - 1:
                x_dict_element_grouping = torch.cat((x_dict_element_grouping, 
                                torch.zeros(1, x_element.shape[1]).to(device)), axis=0)
                element_grouping_batch = torch.cat((element_grouping_batch, 
                                torch.LongTensor([batch_size-1]).to(device)), axis=0)

            x_element_grouping = global_mean_pool(x_dict_element_grouping, element_grouping_batch) 
        else:
            x_element_grouping = torch.zeros_like(x_element).to(device)  

        if 'horizontal_grouping' in x_dict.keys() and x_dict['horizontal_grouping'].shape[0] != 0:
            x_dict_horizontal_grouping = x_dict['horizontal_grouping']
            horizontal_grouping_batch = torch.LongTensor([0] * x_dict['horizontal_grouping'].shape[0])

            # if the constraint type does not exist in the last graph, 
            # then we need to give a hint to let it know that what is 
            # the last index in the batch
            if horizontal_grouping_batch[-1] != batch_size - 1:
                x_dict_horizontal_grouping = torch.cat((x_dict_horizontal_grouping, 
                                torch.zeros(1, x_element.shape[1]).to(device)), axis=0)
                horizontal_grouping_batch = torch.cat((horizontal_grouping_batch, 
                                torch.LongTensor([batch_size-1]).to(device)), axis=0)

            x_horizontal_grouping = global_mean_pool(x_dict_horizontal_grouping, horizontal_grouping_batch) 
        else:
            x_horizontal_grouping = torch.zeros_like(x_element).to(device)  

        if 'vertical_grouping' in x_dict.keys() and x_dict['vertical_grouping'].shape[0] != 0:
            x_dict_vertical_grouping = x_dict['vertical_grouping']
            vertical_grouping_batch = torch.LongTensor([0] * x_dict['vertical_grouping'].shape[0])

            # if the constraint type does not exist in the last graph, 
            # then we need to give a hint to let it know that what is 
            # the last index in the batch
            if vertical_grouping_batch[-1] != batch_size - 1:
                x_dict_vertical_grouping = torch.cat((x_dict_vertical_grouping, 
                                torch.zeros(1, x_element.shape[1]).to(device)), axis=0)
                vertical_grouping_batch = torch.cat((vertical_grouping_batch, 
                                torch.LongTensor([batch_size-1]).to(device)), axis=0)

            x_vertical_grouping = global_mean_pool(x_dict_vertical_grouping, vertical_grouping_batch) 
        else:
            x_vertical_grouping = torch.zeros_like(x_element).to(device)  

        if 'multimodal_grouping' in x_dict.keys() and x_dict['multimodal_grouping'].shape[0] != 0:

            x_dict_multimodal_grouping = x_dict['multimodal_grouping']
            multimodal_grouping_batch = torch.LongTensor([0] * x_dict['multimodal_grouping'].shape[0])

            # if the constraint type does not exist in the last graph, 
            # then we need to give a hint to let it know that what is 
            # the last index in the batch
            if multimodal_grouping_batch[-1] != batch_size - 1:
                x_dict_multimodal_grouping = torch.cat((x_dict_multimodal_grouping, 
                                torch.zeros(1, x_element.shape[1]).to(device)), axis=0)
                multimodal_grouping_batch = torch.cat((multimodal_grouping_batch, 
                                torch.LongTensor([batch_size-1]).to(device)), axis=0)

            x_multimodal_grouping = global_mean_pool(x_dict_multimodal_grouping, multimodal_grouping_batch) 
        else:
            x_multimodal_grouping = torch.zeros_like(x_element).to(device) 

        # self.lin2()
        # print(x_element.shape, x_alignment.shape, x_size.shape,
        #                         x_element_grouping.shape, x_multimodal_grouping)
      
        # weighted average of all types of node embeddings
        x = torch.mean(self.lin(torch.stack([x_element, x_alignment, x_size,
                                x_element_grouping, x_horizontal_grouping, 
                                x_vertical_grouping, x_multimodal_grouping])), dim=0) 



        # if x_dict['alignment'] != []:
        #     x_dict['alignment'] = self.lin2(x_dict['alignment'])
        # else:
        #     x_dict['alignment'] = torch.empty(0).to(device)
        # if x_dict['size'] != []:
        #     x_dict['size'] = self.lin2(x_dict['size'])
        # else:
        #     x_dict['size'] = torch.empty(0).to(device)
        # if x_dict['gap'] != []:
        #     x_dict['gap'] = self.lin2(x_dict['gap'])
        # else:
        #     x_dict['gap'] = torch.empty(0).to(device)

        if 'alignment' in x_dict.keys():
            x_dict_alignment = x_dict['alignment']
        else:
            x_dict_alignment = None

        if 'size' in x_dict.keys():
            x_dict_size = x_dict['size']
        else:
            x_dict_size = None

        if 'element_grouping' in x_dict.keys():
            x_dict_element_grouping = x_dict['element_grouping']
        else:
            x_dict_element_grouping = None

        if 'horizontal_grouping' in x_dict.keys():
            x_dict_horizontal_grouping = x_dict['horizontal_grouping']
        else:
            x_dict_horizontal_grouping = None

        if 'vertical_grouping' in x_dict.keys():
            x_dict_vertical_grouping = x_dict['vertical_grouping']
        else:
            x_dict_vertical_grouping = None

        if 'multimodal_grouping' in x_dict.keys():
            x_dict_multimodal_grouping = x_dict['multimodal_grouping']
        else:
            x_dict_multimodal_grouping = None
        
        return x, x_dict_alignment, x_dict_size, \
                x_dict_element_grouping, \
                x_dict_horizontal_grouping, \
            x_dict_vertical_grouping, x_dict_multimodal_grouping