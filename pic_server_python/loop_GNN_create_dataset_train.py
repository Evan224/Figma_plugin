import torch
import torchvision
#from transformers import BertTokenizer, BertModel
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm
import math
#import pandas as pd
import os, random
import json, glob
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import time, pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

FOLDER = 'figma_data'

# pretrained resnet 
resnet50 = models.resnet50(pretrained=True).to(device)

# if  __name__ == '__main__':
#     FOLDER = 'figma_data'

#     # pretrained resnet 
#     resnet50 = models.resnet50(pretrained=True).to(device)


#     # resnet152 = models.resnet152(pretrained=True).to(device)
#     # modules=list(resnet152.children())[:-1]
#     # resnet152=nn.Sequential(*modules)
#     # for p in resnet152.parameters():
#     #     p.requires_grad = False

#     # Loading the pre-trained BERT model
#     # Embeddings will be derived from
#     # the outputs of this model
#     bert_model = BertModel.from_pretrained('bert-base-uncased',
#                                       output_hidden_states = True,
#                                       ).to(device)

#     # Setting up the tokenizer
#     # This is the same tokenizer that
#     # was used in the model to generate 
#     # embeddings to ensure consistency
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors
    
def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

def add_constraints_to_partial_UI(data, result):
    existing_and_target_ele_ids = [ele['id'] for ele in result['elements']]
    existing_and_target_ele_ids.append(result['target']['id'])
    target_id = result['target']['id']
    
    # generate alignment constraints for partial UIs
    alignment = data['alignment']

    for a_type in alignment.keys():

        del_list = []
        for align in alignment[a_type].keys():
            # we only keep constraints for elements we have
            remove_ids = []
            for ele_id in alignment[a_type][align]:
                if ele_id not in existing_and_target_ele_ids:
                    remove_ids.append(ele_id)
            for ele_id in remove_ids:
                alignment[a_type][align].remove(ele_id)
            if len(alignment[a_type][align]) < 2:
                del_list.append(align)
        for d in del_list:
            del alignment[a_type][d]
    result['alignment'] = alignment

    # generate size constraints for partial UIs
    size = data['size']
    for s_type in size.keys():

        del_list = []
        for si in size[s_type].keys():

            # we only keep constraints for elements we have
            remove_ids = []
            for ele_id in size[s_type][si]:
                if ele_id not in existing_and_target_ele_ids:
                    remove_ids.append(ele_id)
            for ele_id in remove_ids:
                size[s_type][si].remove(ele_id)
            if len(size[s_type][si]) < 2:
                del_list.append(si)
        for d in del_list:
            del size[s_type][d]
    result['size'] = size

    # generate element grouping constraints for partial UIs
    element_grouping = data['element grouping']
    for e_type in element_grouping.keys():

        del_list = []
        for elegroup in element_grouping[e_type].keys():

            # we only keep constraints for elements we have
            remove_ids = []
            for ele_id in element_grouping[e_type][elegroup]:
                if ele_id not in existing_and_target_ele_ids:
                    remove_ids.append(ele_id)
            for ele_id in remove_ids:
                element_grouping[e_type][elegroup].remove(ele_id)
            if len(element_grouping[e_type][elegroup]) < 2:
                del_list.append(elegroup)
        for d in del_list:
            del element_grouping[e_type][d]
    result['element grouping'] = element_grouping


    # generate element grouping constraints for partial UIs
    element_grouping = data['horizontal_groups']
    # for e_type in element_grouping.keys():

    del_list = []
    for elegroup in element_grouping.keys():

        # we only keep constraints for elements we have
        remove_ids = []
        for ele_id in element_grouping[elegroup]:
            if ele_id not in existing_and_target_ele_ids:
                remove_ids.append(ele_id)
        for ele_id in remove_ids:
            element_grouping[elegroup].remove(ele_id)
        if len(element_grouping[elegroup]) < 2:
            del_list.append(elegroup)
    for d in del_list:
        del element_grouping[d]
    result['horizontal_groups'] = element_grouping

    # generate element grouping constraints for partial UIs
    element_grouping = data['vertical_groups']
    # for e_type in element_grouping.keys():

    del_list = []
    for elegroup in element_grouping.keys():

        # we only keep constraints for elements we have
        remove_ids = []
        for ele_id in element_grouping[elegroup]:
            if ele_id not in existing_and_target_ele_ids:
                remove_ids.append(ele_id)
        for ele_id in remove_ids:
            element_grouping[elegroup].remove(ele_id)
        if len(element_grouping[elegroup]) < 2:
            del_list.append(elegroup)
    for d in del_list:
        del element_grouping[d]
    result['vertical_groups'] = element_grouping

    # generate multimodal grouping constraints for partial UIs
    multimodal_grouping = data['multimodal grouping']
    del_keys = []
    for m_type in multimodal_grouping.keys():

        del_list = []
        for multigroup in range(len(multimodal_grouping[m_type])):

            # we only keep constraints for elements we have
            remove_ids = []
            for ele_id in multimodal_grouping[m_type][multigroup]:
                if ele_id not in existing_and_target_ele_ids:
                    remove_ids.append(ele_id)
            for ele_id in remove_ids:
                multimodal_grouping[m_type][multigroup].remove(ele_id)
            if len(multimodal_grouping[m_type][multigroup]) < 2:
                del_list.append(multigroup)
        for dd in range(len(del_list) - 1, -1, -1):
            multimodal_grouping[m_type].pop(del_list[dd])
        if multimodal_grouping[m_type] == []:
            del_keys.append(m_type)
    for d in del_keys:
        del multimodal_grouping[d]
    result['multimodal grouping'] = multimodal_grouping

    return result

    
def get_frequent_texts():
    files = sorted(glob.glob('/home/yuejiang/Documents/Autocompletion/dataset/json_full_UIs/*.json'))

    text_frequency_dict = dict()

    for i in range(len(files)):
        with open(files[i]) as f:
            data = json.load(f)
            for element in data['elements']:
                text = element['text']
                if text not in text_frequency_dict.keys():
                    text_frequency_dict[text] = 1
                else:
                    text_frequency_dict[text] += 1

    frequent_text_list = []
    for key in text_frequency_dict.keys():
        if text_frequency_dict[key] >= 3:
            frequent_text_list.append(key)

    return frequent_text_list


# we have 23 types in total
types = ['Text', 'PageIndicator', 'TextButton', 'Image', 'Icon', 'UpperTaskBar', 
         'EditText', 'Text Button', 'Switch', 'Input', 'CheckedTextView', 'Toolbar', 
         'Multi_Tab', 'Radio Button', 'On/Off Switch', 'Web View', 'Slider', 
         'Pager Indicator', 'List Item', 'Card', 'Spinner', 'Advertisement', 'CheckBox']


TOLERANCE = 0



def process(json_data, ui_name):
    print('====')

    # get frequent text list
    with open("./figma_data/frequent_text.pt", "rb") as fp:  
        frequent_text_list = pickle.load(fp)

    # get frequent text to embedding mapping
    with open("./figma_data/text_to_embed.pt", "rb") as fp:  
        text_to_embed = pickle.load(fp)


    data = json_data


    # read the screenshot and resize to the size in json
    image_name = ui_name
    image_path = './dataset/UI_images/' + image_name + '.jpg'
    image = Image.open(image_path)
    image = image.resize((int(data['width']), int(data['height'])))

    # get width and height of the UI
    ui_width_height = [int(data['width']), int(data['height'])]

    # get element list, the list of (x, y, w, h)
    element_list = []
    text_list = []
    type_list = []
    id_list = []
    id_to_ele_index_map = {} # key: ele id, value: index in the ele list
    for i in range(len(data['elements'])):
        ele = data['elements'][i]
        ele_x = ele['left']
        ele_y = ele['top']
        ele_w = ele['width']
        ele_h = ele['height']
        ele_text = ele['text']
        if ele_text not in frequent_text_list:
            ele_text = ''
        ele_id = ele['id']
        t = ele['type']
        ele_type = [0] * len(types)
        ele_type[types.index(t)] = 1
        element_list.append([ele_x, ele_y, ele_w, ele_h])
        text_list.append(ele_text)
        type_list.append(ele_type)
        id_list.append(ele_id)
        id_to_ele_index_map[ele_id] = i

    # get the target element info
    target = data['target']
    target_x = target['left']
    target_y = target['top']
    target_w = target['width']
    target_h = target['height']
    target_text = target['text']
    if target_text not in frequent_text_list:
        target_text = ''
    target_id = target['id']
    t = target['type']
    ele_type = [0] * len(types)
    ele_type[types.index(t)] = 1
    target_type = ele_type

    # define one-hot alignments (left 0, top 1, right 2, bottom 3)
    # horizontal gridline coords are (0, k)
    # vertical gridline coords are (k, 0)
    # and define element to alignment mapping
    alignment_list = []
    alignment_tags = []
    element_aligns_alignment = [[], []]
    binary_target_alignment_links = []
    alignments = data['alignment']
    align_index = 0
    align_left = alignments['left']
    for x in align_left.keys():
        alignment_list.append([1, 0, 0, 0, 0, 0, float(x), 0])
        alignment_tags.append(['left', float(x)])
        ele_list = align_left[x]
        target_align = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_align = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_aligns_alignment[0].append(index)
                element_aligns_alignment[1].append(align_index)
        binary_target_alignment_links.append(target_align)
        align_index += 1

    align_right = alignments['right']
    for x in align_right.keys():
        alignment_list.append([0, 1, 0, 0, 0, 0, float(x), 0])
        alignment_tags.append(['right', float(x)])
        ele_list = align_right[x]
        target_align = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_align = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_aligns_alignment[0].append(index)
                element_aligns_alignment[1].append(align_index)
        binary_target_alignment_links.append(target_align)
        align_index += 1

    align_vertical_midline = alignments['vertical_midline']
    for x in align_vertical_midline.keys():
        alignment_list.append([0, 0, 1, 0, 0, 0, float(x), 0])
        alignment_tags.append(['vertical_midline', float(x)])
        ele_list = align_vertical_midline[x]
        target_align = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_align = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_aligns_alignment[0].append(index)
                element_aligns_alignment[1].append(align_index)
        binary_target_alignment_links.append(target_align)
        align_index += 1

    align_top = alignments['top']
    for y in align_top.keys():
        alignment_list.append([0, 0, 0, 1, 0, 0, 0, float(y)])
        alignment_tags.append(['top', float(y)])
        ele_list = align_top[y]
        target_align = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_align = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_aligns_alignment[0].append(index)
                element_aligns_alignment[1].append(align_index)
        binary_target_alignment_links.append(target_align)
        align_index += 1

    align_bottom = alignments['bottom']
    for y in align_bottom.keys():
        alignment_list.append([0, 0, 0, 0, 1, 0, 0, float(y)])
        alignment_tags.append(['bottom', float(y)])
        ele_list = align_bottom[y]
        target_align = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_align = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_aligns_alignment[0].append(index)
                element_aligns_alignment[1].append(align_index)
        binary_target_alignment_links.append(target_align)
        align_index += 1

    align_horizontal_midline = alignments['horizontal_midline']
    for y in align_horizontal_midline.keys():
        alignment_list.append([0, 0, 0, 0, 0, 1, 0, float(y)])
        alignment_tags.append(['horizontal_midline', float(y)])
        ele_list = align_horizontal_midline[y]
        target_align = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_align = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_aligns_alignment[0].append(index)
                element_aligns_alignment[1].append(align_index)
        binary_target_alignment_links.append(target_align)
        align_index += 1

    # print(id_to_ele_index_map)
    # print(element_aligns_alignment)
    # print(alignment_list)
    # print(binary_target_alignment_links)

    # add size constraints
    size_list = []
    size_tags = []
    element_has_size = [[], []]
    binary_target_size_links = []
    sizes = data['size']
    size_index = 0
    size_width = sizes['width']
    for x in size_width.keys():
        size_list.append([float(x), 0])
        size_tags.append(['width', float(x)])
        ele_list = size_width[x]
        target_size = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_size = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_has_size[0].append(index)
                element_has_size[1].append(size_index)
        binary_target_size_links.append(target_size)
        size_index += 1

    size_height = sizes['height']
    for y in size_height.keys():
        size_list.append([0, float(y)])
        size_tags.append(['height', float(y)])
        ele_list = size_height[y]
        target_size = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_size = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_has_size[0].append(index)
                element_has_size[1].append(size_index)
        binary_target_size_links.append(target_size)
        size_index += 1

    # print(id_to_ele_index_map)
    # print(element_has_size)
    # print(size_list)
    # print(binary_target_size_links)

    # add element grouping constraints
    element_grouping_list = []
    element_grouping_tags = []
    element_with_element_grouping = [[], []]
    binary_target_element_grouping_links = []
    element_grouping = data['element grouping']
    element_grouping_index = 0
    element_grouping_label_list = []
    element_grouping_vertical = element_grouping['vertical']
    for x in element_grouping_vertical.keys():

        element_grouping_label_list.append(x)
        width = float(x.split('#')[0])
        height = float(x.split('#')[1])
        dist = float(x.split('#')[2])
        element_grouping_tags.append(['vertical', width, height, dist])

        element_grouping_list.append([width, height])
        ele_list = element_grouping_vertical[x]
        target_element_grouping = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_element_grouping = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_with_element_grouping[0].append(index)
                element_with_element_grouping[1].append(element_grouping_index)
        binary_target_element_grouping_links.append(target_element_grouping)
        element_grouping_index += 1

    element_grouping_horizontal = element_grouping['horizontal']
    for x in element_grouping_horizontal.keys():

        element_grouping_label_list.append(x)
        width = float(x.split('#')[0])
        height = float(x.split('#')[1])
        dist = float(x.split('#')[2])
        element_grouping_tags.append(['horizontal', width, height, dist])

        element_grouping_list.append([width, height])
        ele_list = element_grouping_horizontal[x]
        target_element_grouping = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_element_grouping = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_with_element_grouping[0].append(index)
                element_with_element_grouping[1].append(element_grouping_index)
        binary_target_element_grouping_links.append(target_element_grouping)
        element_grouping_index += 1

    # print(id_to_ele_index_map)
    # print(element_with_element_grouping)
    # print(element_grouping_list)
    # print(binary_target_element_grouping_links)

    # add multimodal grouping constraints (at most five)
    multimodal_grouping_list = []
    multimodal_grouping_tags = []
    element_belongs_multimodal_grouping = [[], []]
    binary_target_multimodal_grouping_links = []
    multimodal_grouping = data['multimodal grouping']
    multimodal_grouping_index = 0
    multimodal_grouping_label_list = []
    for x in multimodal_grouping.keys():
        multimodal_grouping_label_list.append(x)
        width = float(x.split('#')[0].split(',')[0][1:])
        height = float(x.split('#')[1].split(',')[0][1:])               
        multimodal_ele_list = multimodal_grouping[x]
        w_list = eval(x.split('#')[0])
        h_list = eval(x.split('#')[1])
        d_list = eval(x.split('#')[2])

        for ele_list in multimodal_ele_list:
            multimodal_grouping_tags.append([w_list, h_list, d_list])
            multimodal_grouping_list.append([width, height])
            target_multimodal_grouping = 0
            for ele_id in ele_list:
                if ele_id == target_id:
                    target_multimodal_grouping = 1
                else:
                    index = id_to_ele_index_map[ele_id]
                    element_belongs_multimodal_grouping[0].append(index)
                    element_belongs_multimodal_grouping[1].append(multimodal_grouping_index)
            binary_target_multimodal_grouping_links.append(target_multimodal_grouping)
            multimodal_grouping_index += 1

    # add horizontal and vertical identical element grouping constraints 
    horizontal_grouping_list = []
    horizontal_grouping_tags = []
    element_hwith_horizontal_grouping = [[], []]
    binary_target_horizontal_grouping_links = []
    horizontal_grouping = data['horizontal_groups']
    horizontal_grouping_index = 0
    horizontal_grouping_label_list = []
    for x in horizontal_grouping.keys():

        horizontal_grouping_label_list.append(x)
        hg_top = float(x.split('/')[0])
        hg_bottom = float(x.split('/')[1])
        if horizontal_grouping[x][0] != target_id:
            ele = data['elements'][id_to_ele_index_map[horizontal_grouping[x][0]]]
        else:
            ele = data['elements'][id_to_ele_index_map[horizontal_grouping[x][1]]]
        hg_width = ele['width']
        hg_height = ele['height']
        horizontal_grouping_tags.append(['horizontal', hg_top, hg_bottom, hg_width, hg_height])

        horizontal_grouping_list.append([hg_top, hg_bottom, hg_width, hg_height])
        ele_list = horizontal_grouping[x]
        target_horizontal_grouping = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_horizontal_grouping = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_hwith_horizontal_grouping[0].append(index)
                element_hwith_horizontal_grouping[1].append(horizontal_grouping_index)
        binary_target_horizontal_grouping_links.append(target_horizontal_grouping)
        horizontal_grouping_index += 1

    vertical_grouping_list = []
    vertical_grouping_tags = []
    element_vwith_vertical_grouping = [[], []]
    binary_target_vertical_grouping_links = []
    vertical_grouping = data['vertical_groups']
    vertical_grouping_index = 0
    vertical_grouping_label_list = []
    for x in vertical_grouping.keys():

        vertical_grouping_label_list.append(x)
        vg_left = float(x.split('/')[0])
        vg_right = float(x.split('/')[1])
        # print(id_to_ele_index_map[vertical_grouping[x][0]], len(data['elements']))
        if vertical_grouping[x][0] != target_id:
            ele = data['elements'][id_to_ele_index_map[vertical_grouping[x][0]]]
        else:
            ele = data['elements'][id_to_ele_index_map[vertical_grouping[x][1]]]
        vg_width = ele['width']
        vg_height = ele['height']
        vertical_grouping_tags.append(['vertical', vg_left, vg_right, vg_width, vg_height])

        vertical_grouping_list.append([vg_left, vg_right, vg_width, vg_height])
        ele_list = vertical_grouping[x]
        target_vertical_grouping = 0
        for ele_id in ele_list:
            if ele_id == target_id:
                target_vertical_grouping = 1
            else:
                index = id_to_ele_index_map[ele_id]
                element_vwith_vertical_grouping[0].append(index)
                element_vwith_vertical_grouping[1].append(vertical_grouping_index)
        binary_target_vertical_grouping_links.append(target_vertical_grouping)
        vertical_grouping_index += 1

    # print(id_to_ele_index_map)
    # print(element_belongs_multimodal_grouping)
    # print(multimodal_grouping_list)
    # print(binary_target_multimodal_grouping_links)

    # create a heterogeneous graph
    data = HeteroData()
    data['element'].x = torch.tensor(element_list).type(torch.FloatTensor).to(device)
    # data['element'].y = ui_width_height

    data['alignment'].x = torch.tensor(alignment_list).type(torch.FloatTensor).to(device)
    data['alignment'].y = alignment_tags

    data['size'].x = torch.tensor(size_list).type(torch.FloatTensor).to(device)
    data['size'].y = size_tags

    data['element_grouping'].x = torch.tensor(element_grouping_list).type(torch.FloatTensor).to(device)
    data['element_grouping'].y = element_grouping_tags

    data['multimodal_grouping'].x = torch.tensor(multimodal_grouping_list).type(torch.FloatTensor).to(device)
    data['multimodal_grouping'].y = multimodal_grouping_tags

    data['horizontal_grouping'].x = torch.tensor(horizontal_grouping_list).type(torch.FloatTensor).to(device)
    data['horizontal_grouping'].y = horizontal_grouping_tags

    data['vertical_grouping'].x = torch.tensor(vertical_grouping_list).type(torch.FloatTensor).to(device)
    data['vertical_grouping'].y = vertical_grouping_tags

    data['element', 'aligns', 'alignment'].edge_index = torch.tensor(element_aligns_alignment).type(torch.FloatTensor).to(device)
    data['element', 'has', 'size'].edge_index = torch.tensor(element_has_size).type(torch.FloatTensor).to(device)
    data['element', 'with', 'element_grouping'].edge_index = torch.tensor(element_with_element_grouping).type(torch.FloatTensor).to(device)
    data['element', 'hwith', 'horizontal_grouping'].edge_index = torch.tensor(element_hwith_horizontal_grouping).type(torch.FloatTensor).to(device)
    data['element', 'vwith', 'vertical_grouping'].edge_index = torch.tensor(element_vwith_vertical_grouping).type(torch.FloatTensor).to(device)
    data['element', 'with', 'element_grouping'].edge_index = torch.tensor(element_with_element_grouping).type(torch.FloatTensor).to(device)
    data['element', 'belongs', 'multimodal_grouping'].edge_index = torch.tensor(element_belongs_multimodal_grouping).type(torch.FloatTensor).to(device)
    data['element'].y = torch.tensor([[target_x, target_y, target_w, target_h]]).type(torch.FloatTensor).to(device)

    # add reverse edges
    data = ToUndirected()(data)


    # filter out cases with invalid texts or images
    # try:
    if True:
        text_embedding_list = []
        for i in range(len(text_list)):
            text = text_list[i]

            # # get the bert model ready
            # tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
            # text_embedding = get_bert_embeddings(tokens_tensor.to(device), segments_tensors.to(device), bert_model)

            # # CLS embedding can be seen as the sentence embedding
            if i == 0:
                text_embedding_list = text_to_embed[text].unsqueeze(0)
            else:
                text_embedding_list = torch.concat((text_embedding_list, text_to_embed[text].unsqueeze(0)), axis = 0)
        # print(text_embedding_list.shape)

        # text_embedding_list = torch.tensor(text_embedding_list)

        # # get the bert model ready
        # tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(target_text, tokenizer)
        # text_embedding = torch.tensor(get_bert_embeddings(tokens_tensor.to(device), segments_tensors.to(device), bert_model))

        # # CLS embedding can be seen as the sentence embedding
        target_text_embedding = text_to_embed[target_text]


        image_embedding_list = None
    
        for element in element_list:

            left = int(element[0])
            top = int(element[1])
            width = int((element[2]))
            height = int((element[3]))
            right = left + width
            bottom = top + height

            element_image = image.crop((left, top, right, bottom))

            # element_image = torchvision.transforms.Resize((224, 224))(element_image) / 255.
            # image_embedding = resnet152(element_image)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) ])
            element_image = transform(element_image).unsqueeze(0).to(device)


            image_embedding = resnet50(element_image)[0,:]
            if image_embedding_list is None:
                image_embedding_list = image_embedding.unsqueeze(0)
            else:
                image_embedding_list = torch.concat((image_embedding_list, image_embedding.unsqueeze(0)), axis=0)

        left = int(target_x)
        top = int(target_y)
        width = int(target_w)
        height = int(target_h)
        right = left + width
        bottom = top + height

        element_image = image.crop((left, top, right, bottom))

        # element_image = torchvision.transforms.Resize((224, 224))(element_image) / 255.
        # image_embedding = resnet152(element_image)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]) ])
        element_image = transform(element_image).unsqueeze(0).to(device)

        image_embedding = resnet50(element_image)[0,:]
        target_image_embedding = image_embedding
    
    # save the graph data
    data = data.clone().detach().requires_grad_(False)
    ui_width_height = torch.tensor(ui_width_height).clone().detach().requires_grad_(False)
    ele_text_embedding = text_embedding_list.clone().detach().requires_grad_(False)
    ele_image_embedding = image_embedding_list.clone().detach().requires_grad_(False)
    ele_type = torch.tensor(type_list).clone().detach().requires_grad_(False)
    target_text_embedding = target_text_embedding.clone().detach().requires_grad_(False)
    target_image_embedding = target_image_embedding.clone().detach().requires_grad_(False)
    target_type = torch.tensor(target_type).clone().detach().requires_grad_(False)
    binary_target_alignment_links = torch.tensor(binary_target_alignment_links).clone().detach().requires_grad_(False)
    binary_target_size_links = torch.tensor(binary_target_size_links).clone().detach().requires_grad_(False)
    binary_target_element_grouping_links = torch.tensor(binary_target_element_grouping_links).clone().detach().requires_grad_(False)
    binary_target_horizontal_grouping_links = torch.tensor(binary_target_horizontal_grouping_links).clone().detach().requires_grad_(False)
    binary_target_vertical_grouping_links = torch.tensor(binary_target_vertical_grouping_links).clone().detach().requires_grad_(False)
    binary_target_multimodal_grouping_links = torch.tensor(binary_target_multimodal_grouping_links).clone().detach().requires_grad_(False)

    return data, ui_width_height, ele_text_embedding, ele_image_embedding, ele_type, target_text_embedding, target_image_embedding, \
            target_type, binary_target_alignment_links, binary_target_size_links, binary_target_element_grouping_links, \
            binary_target_horizontal_grouping_links, binary_target_vertical_grouping_links, binary_target_multimodal_grouping_links

            



class PatternDataset(Dataset):
    def __init__(self, root, json_data, ui_name, transform=None, pre_transform=None, pre_filter=None):

        super().__init__(root, json_data, ui_name)
        self.root = root
        self.json_data = json_data
        self.ui_name = ui_name

    @property
    def raw_file_names(self):
        # print(sorted(glob.glob('./' + self.root + '/raw/*.json')))
        # exit()
        
        # return sorted(glob.glob('./{}/raw/*.json'.format(self.root)))
        return ['figma_data/raw/./figma_data/raw/Android_56_4.json', self.root]

    @property
    def processed_file_names(self):
        folders = glob.glob('./{}/processed/*'.format(self.root))
        if folders == []:
            return folders

        try:
            folders.remove('./{}/processed/pre_filter.pt'.format(self.root))
            folders.remove('./{}/processed/pre_transform.pt'.format(self.root))
        except:
            pass
        return folders

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        print('====')

        print(self.ui_name)
        exit()
        

        output_dir = './' + self.root + '/processed/'

        # get frequent text list
        with open("./figma_data/frequent_text.pt", "rb") as fp:  
            frequent_text_list = pickle.load(fp)

        # get frequent text to embedding mapping
        with open("./figma_data/text_to_embed.pt", "rb") as fp:  
            text_to_embed = pickle.load(fp)

        # process the json files in the raw folder
        count = 0
        print(self.raw_paths)
        for file, json_data, ui_name in self.raw_paths:
            file = '/'.join(file.split('/')[2:])
            if count % 1000 == 0:
                print(count)
            count += 1
            folder_name = file.split('/')[-1].split('.')[0]
            output_folder = output_dir + folder_name
            os.makedirs(output_folder, exist_ok=True)
            with open(file) as f:
                data = json.load(f)

            print(file)
            print('+++', folder_name, output_folder, ui_name)
            exit()

            # read the screenshot and resize to the size in json
            image_name = '_'.join(folder_name.split('_')[:-1])
            image_path = './dataset/UI_images/' + image_name + '.jpg'
            image = Image.open(image_path)
            image = image.resize((int(data['width']), int(data['height'])))
        
            # get width and height of the UI
            ui_width_height = [int(data['width']), int(data['height'])]

            # get element list, the list of (x, y, w, h)
            element_list = []
            text_list = []
            type_list = []
            id_list = []
            id_to_ele_index_map = {} # key: ele id, value: index in the ele list
            for i in range(len(data['elements'])):
                ele = data['elements'][i]
                ele_x = ele['left']
                ele_y = ele['top']
                ele_w = ele['width']
                ele_h = ele['height']
                ele_text = ele['text']
                if ele_text not in frequent_text_list:
                    ele_text = ''
                ele_id = ele['id']
                t = ele['type']
                ele_type = [0] * len(types)
                ele_type[types.index(t)] = 1
                element_list.append([ele_x, ele_y, ele_w, ele_h])
                text_list.append(ele_text)
                type_list.append(ele_type)
                id_list.append(ele_id)
                id_to_ele_index_map[ele_id] = i

            # get the target element info
            target = data['target']
            target_x = target['left']
            target_y = target['top']
            target_w = target['width']
            target_h = target['height']
            target_text = target['text']
            if target_text not in frequent_text_list:
                target_text = ''
            target_id = target['id']
            t = target['type']
            ele_type = [0] * len(types)
            ele_type[types.index(t)] = 1
            target_type = ele_type

            # define one-hot alignments (left 0, top 1, right 2, bottom 3)
            # horizontal gridline coords are (0, k)
            # vertical gridline coords are (k, 0)
            # and define element to alignment mapping
            alignment_list = []
            alignment_tags = []
            element_aligns_alignment = [[], []]
            binary_target_alignment_links = []
            alignments = data['alignment']
            align_index = 0
            align_left = alignments['left']
            for x in align_left.keys():
                alignment_list.append([1, 0, 0, 0, 0, 0, float(x), 0])
                alignment_tags.append(['left', float(x)])
                ele_list = align_left[x]
                target_align = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_align = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_aligns_alignment[0].append(index)
                        element_aligns_alignment[1].append(align_index)
                binary_target_alignment_links.append(target_align)
                align_index += 1

            align_right = alignments['right']
            for x in align_right.keys():
                alignment_list.append([0, 1, 0, 0, 0, 0, float(x), 0])
                alignment_tags.append(['right', float(x)])
                ele_list = align_right[x]
                target_align = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_align = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_aligns_alignment[0].append(index)
                        element_aligns_alignment[1].append(align_index)
                binary_target_alignment_links.append(target_align)
                align_index += 1

            align_vertical_midline = alignments['vertical_midline']
            for x in align_vertical_midline.keys():
                alignment_list.append([0, 0, 1, 0, 0, 0, float(x), 0])
                alignment_tags.append(['vertical_midline', float(x)])
                ele_list = align_vertical_midline[x]
                target_align = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_align = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_aligns_alignment[0].append(index)
                        element_aligns_alignment[1].append(align_index)
                binary_target_alignment_links.append(target_align)
                align_index += 1

            align_top = alignments['top']
            for y in align_top.keys():
                alignment_list.append([0, 0, 0, 1, 0, 0, 0, float(y)])
                alignment_tags.append(['top', float(y)])
                ele_list = align_top[y]
                target_align = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_align = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_aligns_alignment[0].append(index)
                        element_aligns_alignment[1].append(align_index)
                binary_target_alignment_links.append(target_align)
                align_index += 1

            align_bottom = alignments['bottom']
            for y in align_bottom.keys():
                alignment_list.append([0, 0, 0, 0, 1, 0, 0, float(y)])
                alignment_tags.append(['bottom', float(y)])
                ele_list = align_bottom[y]
                target_align = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_align = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_aligns_alignment[0].append(index)
                        element_aligns_alignment[1].append(align_index)
                binary_target_alignment_links.append(target_align)
                align_index += 1

            align_horizontal_midline = alignments['horizontal_midline']
            for y in align_horizontal_midline.keys():
                alignment_list.append([0, 0, 0, 0, 0, 1, 0, float(y)])
                alignment_tags.append(['horizontal_midline', float(y)])
                ele_list = align_horizontal_midline[y]
                target_align = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_align = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_aligns_alignment[0].append(index)
                        element_aligns_alignment[1].append(align_index)
                binary_target_alignment_links.append(target_align)
                align_index += 1

            # print(id_to_ele_index_map)
            # print(element_aligns_alignment)
            # print(alignment_list)
            # print(binary_target_alignment_links)

            # add size constraints
            size_list = []
            size_tags = []
            element_has_size = [[], []]
            binary_target_size_links = []
            sizes = data['size']
            size_index = 0
            size_width = sizes['width']
            for x in size_width.keys():
                size_list.append([float(x), 0])
                size_tags.append(['width', float(x)])
                ele_list = size_width[x]
                target_size = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_size = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_has_size[0].append(index)
                        element_has_size[1].append(size_index)
                binary_target_size_links.append(target_size)
                size_index += 1

            size_height = sizes['height']
            for y in size_height.keys():
                size_list.append([0, float(y)])
                size_tags.append(['height', float(y)])
                ele_list = size_height[y]
                target_size = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_size = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_has_size[0].append(index)
                        element_has_size[1].append(size_index)
                binary_target_size_links.append(target_size)
                size_index += 1

            # print(id_to_ele_index_map)
            # print(element_has_size)
            # print(size_list)
            # print(binary_target_size_links)

            # add element grouping constraints
            element_grouping_list = []
            element_grouping_tags = []
            element_with_element_grouping = [[], []]
            binary_target_element_grouping_links = []
            element_grouping = data['element grouping']
            element_grouping_index = 0
            element_grouping_label_list = []
            element_grouping_vertical = element_grouping['vertical']
            for x in element_grouping_vertical.keys():

                element_grouping_label_list.append(x)
                width = float(x.split('#')[0])
                height = float(x.split('#')[1])
                dist = float(x.split('#')[2])
                element_grouping_tags.append(['vertical', width, height, dist])

                element_grouping_list.append([width, height])
                ele_list = element_grouping_vertical[x]
                target_element_grouping = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_element_grouping = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_with_element_grouping[0].append(index)
                        element_with_element_grouping[1].append(element_grouping_index)
                binary_target_element_grouping_links.append(target_element_grouping)
                element_grouping_index += 1

            element_grouping_horizontal = element_grouping['horizontal']
            for x in element_grouping_horizontal.keys():

                element_grouping_label_list.append(x)
                width = float(x.split('#')[0])
                height = float(x.split('#')[1])
                dist = float(x.split('#')[2])
                element_grouping_tags.append(['horizontal', width, height, dist])

                element_grouping_list.append([width, height])
                ele_list = element_grouping_horizontal[x]
                target_element_grouping = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_element_grouping = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_with_element_grouping[0].append(index)
                        element_with_element_grouping[1].append(element_grouping_index)
                binary_target_element_grouping_links.append(target_element_grouping)
                element_grouping_index += 1

            # print(id_to_ele_index_map)
            # print(element_with_element_grouping)
            # print(element_grouping_list)
            # print(binary_target_element_grouping_links)

            # add multimodal grouping constraints (at most five)
            multimodal_grouping_list = []
            multimodal_grouping_tags = []
            element_belongs_multimodal_grouping = [[], []]
            binary_target_multimodal_grouping_links = []
            multimodal_grouping = data['multimodal grouping']
            multimodal_grouping_index = 0
            multimodal_grouping_label_list = []
            for x in multimodal_grouping.keys():
                multimodal_grouping_label_list.append(x)
                width = float(x.split('#')[0].split(',')[0][1:])
                height = float(x.split('#')[1].split(',')[0][1:])               
                multimodal_ele_list = multimodal_grouping[x]
                w_list = eval(x.split('#')[0])
                h_list = eval(x.split('#')[1])
                d_list = eval(x.split('#')[2])

                for ele_list in multimodal_ele_list:
                    multimodal_grouping_tags.append([w_list, h_list, d_list])
                    multimodal_grouping_list.append([width, height])
                    target_multimodal_grouping = 0
                    for ele_id in ele_list:
                        if ele_id == target_id:
                            target_multimodal_grouping = 1
                        else:
                            index = id_to_ele_index_map[ele_id]
                            element_belongs_multimodal_grouping[0].append(index)
                            element_belongs_multimodal_grouping[1].append(multimodal_grouping_index)
                    binary_target_multimodal_grouping_links.append(target_multimodal_grouping)
                    multimodal_grouping_index += 1

            # add horizontal and vertical identical element grouping constraints 
            horizontal_grouping_list = []
            horizontal_grouping_tags = []
            element_hwith_horizontal_grouping = [[], []]
            binary_target_horizontal_grouping_links = []
            horizontal_grouping = data['horizontal_groups']
            horizontal_grouping_index = 0
            horizontal_grouping_label_list = []
            for x in horizontal_grouping.keys():

                horizontal_grouping_label_list.append(x)
                hg_top = float(x.split('/')[0])
                hg_bottom = float(x.split('/')[1])
                if horizontal_grouping[x][0] != target_id:
                    ele = data['elements'][id_to_ele_index_map[horizontal_grouping[x][0]]]
                else:
                    ele = data['elements'][id_to_ele_index_map[horizontal_grouping[x][1]]]
                hg_width = ele['width']
                hg_height = ele['height']
                horizontal_grouping_tags.append(['horizontal', hg_top, hg_bottom, hg_width, hg_height])

                horizontal_grouping_list.append([hg_top, hg_bottom, hg_width, hg_height])
                ele_list = horizontal_grouping[x]
                target_horizontal_grouping = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_horizontal_grouping = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_hwith_horizontal_grouping[0].append(index)
                        element_hwith_horizontal_grouping[1].append(horizontal_grouping_index)
                binary_target_horizontal_grouping_links.append(target_horizontal_grouping)
                horizontal_grouping_index += 1

            vertical_grouping_list = []
            vertical_grouping_tags = []
            element_vwith_vertical_grouping = [[], []]
            binary_target_vertical_grouping_links = []
            vertical_grouping = data['vertical_groups']
            vertical_grouping_index = 0
            vertical_grouping_label_list = []
            for x in vertical_grouping.keys():

                vertical_grouping_label_list.append(x)
                vg_left = float(x.split('/')[0])
                vg_right = float(x.split('/')[1])
                # print(id_to_ele_index_map[vertical_grouping[x][0]], len(data['elements']))
                if vertical_grouping[x][0] != target_id:
                    ele = data['elements'][id_to_ele_index_map[vertical_grouping[x][0]]]
                else:
                    ele = data['elements'][id_to_ele_index_map[vertical_grouping[x][1]]]
                vg_width = ele['width']
                vg_height = ele['height']
                vertical_grouping_tags.append(['vertical', vg_left, vg_right, vg_width, vg_height])

                vertical_grouping_list.append([vg_left, vg_right, vg_width, vg_height])
                ele_list = vertical_grouping[x]
                target_vertical_grouping = 0
                for ele_id in ele_list:
                    if ele_id == target_id:
                        target_vertical_grouping = 1
                    else:
                        index = id_to_ele_index_map[ele_id]
                        element_vwith_vertical_grouping[0].append(index)
                        element_vwith_vertical_grouping[1].append(vertical_grouping_index)
                binary_target_vertical_grouping_links.append(target_vertical_grouping)
                vertical_grouping_index += 1

            # print(id_to_ele_index_map)
            # print(element_belongs_multimodal_grouping)
            # print(multimodal_grouping_list)
            # print(binary_target_multimodal_grouping_links)

            # create a heterogeneous graph
            data = HeteroData()
            data['element'].x = torch.tensor(element_list).type(torch.FloatTensor).to(device)
            # data['element'].y = ui_width_height

            data['alignment'].x = torch.tensor(alignment_list).type(torch.FloatTensor).to(device)
            data['alignment'].y = alignment_tags

            data['size'].x = torch.tensor(size_list).type(torch.FloatTensor).to(device)
            data['size'].y = size_tags

            data['element_grouping'].x = torch.tensor(element_grouping_list).type(torch.FloatTensor).to(device)
            data['element_grouping'].y = element_grouping_tags

            data['multimodal_grouping'].x = torch.tensor(multimodal_grouping_list).type(torch.FloatTensor).to(device)
            data['multimodal_grouping'].y = multimodal_grouping_tags

            data['horizontal_grouping'].x = torch.tensor(horizontal_grouping_list).type(torch.FloatTensor).to(device)
            data['horizontal_grouping'].y = horizontal_grouping_tags

            data['vertical_grouping'].x = torch.tensor(vertical_grouping_list).type(torch.FloatTensor).to(device)
            data['vertical_grouping'].y = vertical_grouping_tags

            data['element', 'aligns', 'alignment'].edge_index = torch.tensor(element_aligns_alignment).type(torch.FloatTensor).to(device)
            data['element', 'has', 'size'].edge_index = torch.tensor(element_has_size).type(torch.FloatTensor).to(device)
            data['element', 'with', 'element_grouping'].edge_index = torch.tensor(element_with_element_grouping).type(torch.FloatTensor).to(device)
            data['element', 'hwith', 'horizontal_grouping'].edge_index = torch.tensor(element_hwith_horizontal_grouping).type(torch.FloatTensor).to(device)
            data['element', 'vwith', 'vertical_grouping'].edge_index = torch.tensor(element_vwith_vertical_grouping).type(torch.FloatTensor).to(device)
            data['element', 'with', 'element_grouping'].edge_index = torch.tensor(element_with_element_grouping).type(torch.FloatTensor).to(device)
            data['element', 'belongs', 'multimodal_grouping'].edge_index = torch.tensor(element_belongs_multimodal_grouping).type(torch.FloatTensor).to(device)
            data['element'].y = torch.tensor([[target_x, target_y, target_w, target_h]]).type(torch.FloatTensor).to(device)

            # add reverse edges
            data = ToUndirected()(data)


            # filter out cases with invalid texts or images
            # try:
            if True:
                text_embedding_list = []
                for i in range(len(text_list)):
                    text = text_list[i]

                    # # get the bert model ready
                    # tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
                    # text_embedding = get_bert_embeddings(tokens_tensor.to(device), segments_tensors.to(device), bert_model)

                    # # CLS embedding can be seen as the sentence embedding
                    if i == 0:
                        text_embedding_list = text_to_embed[text].unsqueeze(0)
                    else:
                        text_embedding_list = torch.concat((text_embedding_list, text_to_embed[text].unsqueeze(0)), axis = 0)
                # print(text_embedding_list.shape)

                # text_embedding_list = torch.tensor(text_embedding_list)

                # # get the bert model ready
                # tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(target_text, tokenizer)
                # text_embedding = torch.tensor(get_bert_embeddings(tokens_tensor.to(device), segments_tensors.to(device), bert_model))

                # # CLS embedding can be seen as the sentence embedding
                target_text_embedding = text_to_embed[target_text]


                image_embedding_list = None
            
                for element in element_list:

                    left = int(element[0])
                    top = int(element[1])
                    width = int((element[2]))
                    height = int((element[3]))
                    right = left + width
                    bottom = top + height

                    element_image = image.crop((left, top, right, bottom))

                    # element_image = torchvision.transforms.Resize((224, 224))(element_image) / 255.
                    # image_embedding = resnet152(element_image)
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)), 
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) ])
                    element_image = transform(element_image).unsqueeze(0).to(device)


                    image_embedding = resnet50(element_image)[0,:]
                    if image_embedding_list is None:
                        image_embedding_list = image_embedding.unsqueeze(0)
                    else:
                        image_embedding_list = torch.concat((image_embedding_list, image_embedding.unsqueeze(0)), axis=0)

                left = int(target_x)
                top = int(target_y)
                width = int(target_w)
                height = int(target_h)
                right = left + width
                bottom = top + height

                element_image = image.crop((left, top, right, bottom))

                # element_image = torchvision.transforms.Resize((224, 224))(element_image) / 255.
                # image_embedding = resnet152(element_image)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]) ])
                element_image = transform(element_image).unsqueeze(0).to(device)

                image_embedding = resnet50(element_image)[0,:]
                target_image_embedding = image_embedding
            # except:
            #     exit()
            #     continue


            # save the graph data
            data = data.clone().detach().requires_grad_(False)
            ui_width_height = torch.tensor(ui_width_height).clone().detach().requires_grad_(False)
            ele_text_embedding = text_embedding_list.clone().detach().requires_grad_(False)
            ele_image_embedding = image_embedding_list.clone().detach().requires_grad_(False)
            ele_type = torch.tensor(type_list).clone().detach().requires_grad_(False)
            target_text_embedding = target_text_embedding.clone().detach().requires_grad_(False)
            target_image_embedding = target_image_embedding.clone().detach().requires_grad_(False)
            target_type = torch.tensor(target_type).clone().detach().requires_grad_(False)
            binary_target_alignment_links = torch.tensor(binary_target_alignment_links).clone().detach().requires_grad_(False)
            binary_target_size_links = torch.tensor(binary_target_size_links).clone().detach().requires_grad_(False)
            binary_target_element_grouping_links = torch.tensor(binary_target_element_grouping_links).clone().detach().requires_grad_(False)
            binary_target_horizontal_grouping_links = torch.tensor(binary_target_horizontal_grouping_links).clone().detach().requires_grad_(False)
            binary_target_vertical_grouping_links = torch.tensor(binary_target_vertical_grouping_links).clone().detach().requires_grad_(False)
            binary_target_multimodal_grouping_links = torch.tensor(binary_target_multimodal_grouping_links).clone().detach().requires_grad_(False)




    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        folder = self.processed_file_names[idx]
        file_basename = folder.split('/')[-1]
        data = torch.load(folder + '/data.pt', map_location='cpu')
        # with open(file) as f:
        #         data = json.load(f)
        # ele_text_embedding = torch.load(folder + '/ele_text_embedding.pt', map_location='cpu')
        # ele_image_embedding = torch.load(folder + '/ele_image_embedding.pt', map_location='cpu')
        # ele_type = torch.load(folder + '/ele_type.pt', map_location='cpu')
        # target_text_embedding = torch.load(folder + '/target_text_embedding.pt', map_location='cpu')
        # target_image_embedding = torch.load(folder + '/target_image_embedding.pt', map_location='cpu')
        # target_type = torch.load(folder + '/target_type.pt', map_location='cpu')

        # binary_target_alignment_links = torch.zeros(50)
        # target_alignment_links = torch.load(folder + '/binary_target_alignment_links.pt', map_location='cpu')
        # binary_target_alignment_links[:target_alignment_links.shape[0]] += target_alignment_links

        # binary_target_size_links = torch.zeros(50)
        # target_size_links = torch.load(folder + '/binary_target_size_links.pt', map_location='cpu')
        # binary_target_size_links[:target_size_links.shape[0]] += target_size_links

        # binary_target_element_grouping_links = torch.zeros(50)
        # target_element_grouping_links = torch.load(folder + '/binary_target_element_grouping_links.pt', map_location='cpu')
        # binary_target_element_grouping_links[:target_element_grouping_links.shape[0]] += target_element_grouping_links

        # binary_target_multimodal_grouping_links = torch.zeros(50)
        # target_multimodal_grouping_links = torch.load(folder + '/binary_target_multimodal_grouping_links.pt', map_location='cpu')
        # binary_target_multimodal_grouping_links[:target_multimodal_grouping_links.shape[0]] += target_multimodal_grouping_links

        return file_basename, data, folder


#if  __name__ == '__main__':
 #   dataset = PatternDataset(root='./' + FOLDER + '/')



    # # # NOTE: have to comment this part before running gnn_models_enrico.py
    # torch.save(torch.tensor(binary_target_alignment_links_all).type(torch.FloatTensor), 
    #                 './' + FOLDER + '/processed/19/binary_target_alignment_links.pt')
    # torch.save(torch.tensor(binary_target_size_links_all).type(torch.FloatTensor), 
    #                 './' + FOLDER + '/processed/19/binary_target_size_links.pt')
    # torch.save(torch.tensor(binary_target_gap_links_all).type(torch.FloatTensor), 
    #                 './' + FOLDER + '/processed/19/binary_target_gap_links.pt')

