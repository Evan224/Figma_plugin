from loop_GNN_create_dataset_train import *
from torch_geometric.loader import DataLoader
from models import *
import copy


def figma_fn(orig_json_data, ui_name):

    BATCH_SIZE = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flag= False

    #FOLDER_NAME = './figma_data'
    # FOLDER_NAME = './GNN_dataset_full'

    # pretrained resnet 
    resnet50 = models.resnet50(pretrained=True).to(device)

    

    
    #dataset = PatternDataset(root=FOLDER_NAME, json_data=json_data, ui_name='Android_56')
    torch.manual_seed(12345)
    # train_dataset = dataset[:]
    # test_dataset = dataset[:300]
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # create models
    GNN = HeteroGNN(hidden_channels=512, out_channels=2536, num_layers=5)
    # MLP1 = MLP(input_channels=2, hidden_channels=64, out_channels=64)
    MLP2 = MLP(input_channels=2536 * 2 - (2536 - 2280), hidden_channels=512, out_channels=4)############
    # MLP2 = MLP(input_channels=2864, hidden_channels=256, out_channels=4)
    # CellClass = Classification(input_channels=64 * 3, hidden_channels=64, out_channels=1)
    # AlignmentClass = Classification(input_channels=2864 * 3, hidden_channels=256, out_channels=1)
    # SizeClass = Classification(input_channels=2864 * 3, hidden_channels=256, out_channels=1)
    # GapClass = Classification(input_channels=2864 * 3, hidden_channels=256, out_channels=1)
    AlignmentClass = Classification(input_channels=2536 * 3 - (2536 - 2280), hidden_channels=512, out_channels=1)
    SizeClass = Classification(input_channels=2536 * 3 - (2536 - 2280), hidden_channels=512, out_channels=1)
    HorizontalGroupingClass = Classification(input_channels=2536 * 3 - (2536 - 2280), hidden_channels=512, out_channels=1)
    VerticalGroupingClass = Classification(input_channels=2536 * 3 - (2536 - 2280), hidden_channels=512, out_channels=1)

    ElementGroupingClass = Classification(input_channels=2536 * 3 - (2536 - 2280), hidden_channels=512, out_channels=1)
    MultimodalGroupingClass = Classification(input_channels=2536 * 3 - (2536 - 2280), hidden_channels=512, out_channels=1)

    PosEmbed = PositionEmbedding()
    TypeEmbed = TypeEmbedding()
    SizeEmbed = SizeEmbedding()

    # # # load models
    GNN = torch.load('./checkpoints/1/130_GNN.pt', map_location='cpu')
    # MLP1 = torch.load('./checkpoints_pattern/enrico_advanced/best_MLP1.pt')
    MLP2 = torch.load('./checkpoints/1/130_MLP2.pt', map_location='cpu')
    # CellClass = torch.load('./checkpoints_pattern/9/3000_CellClass.pt')
    AlignmentClass = torch.load('./checkpoints/1/130_AlignmentClass.pt', map_location='cpu')
    SizeClass = torch.load('./checkpoints/1/130_SizeClass.pt', map_location='cpu')
    ElementGroupingClass = torch.load('./checkpoints/1/130_ElementGroupingClass.pt', map_location='cpu')
    HorzontalGroupingClass = torch.load('./checkpoints/1/130_HorizontalGroupingClass.pt', map_location='cpu')
    VerticalGroupingClass = torch.load('./checkpoints/1/130_VerticalGroupingClass.pt', map_location='cpu')

    MultimodalGroupingClass = torch.load('./checkpoints/1/130_MultimodalGroupingClass.pt', map_location='cpu')
    PosEmbed = torch.load('./checkpoints/1/130_PosEmbed.pt', map_location='cpu')
    SizeEmbed = torch.load('./checkpoints/1/130_SizeEmbed.pt', map_location='cpu')
    TypeEmbed = torch.load('./checkpoints/1/130_TypeEmbed.pt', map_location='cpu')
    GNN.to(device)
    # MLP1.to(device)
    MLP2.to(device)
    # CellClass.to(device)
    AlignmentClass.to(device)
    SizeClass.to(device)
    ElementGroupingClass.to(device)
    HorizontalGroupingClass.to(device)
    VerticalGroupingClass.to(device)
    MultimodalGroupingClass.to(device)
    PosEmbed.to(device)
    TypeEmbed.to(device)
    SizeEmbed.to(device)


    def check_overlap(target_element, element_list):
        left = target_element['left']
        right = target_element['right']
        top = target_element['top']
        bottom = target_element['bottom']
        overlap = False

      #  # check whether the target is out of screen
       # if left < 0 or right > 1440 or top < 0 or bottom > 2560 - 300:
        #    overlap = True

        # check overlapping with other elements
        for ele in element_list:
            if not (right <= ele['left'] \
                    or left >= ele['right'] \
                    or bottom <= ele['top'] \
                    or top >= ele['bottom']):
                overlap = True
        return overlap


    print('end load...')


    #raw_orig_path = './{}/raw/'.format(FOLDER_NAME)
    # raw_result_path = '/home/yuejiang/Documents/Autocompletion/{}/raw_result/'.format(FOLDER_NAME)

    # raw_result_path_high = '/home/yuejiang/Documents/Autocompletion/{}/raw_result/high/'.format(FOLDER_NAME)
    # raw_result_path_median = '/home/yuejiang/Documents/Autocompletion/{}/raw_result/median/'.format(FOLDER_NAME)
    # raw_result_path_low = '/home/yuejiang/Documents/Autocompletion/{}/raw_result/low/'.format(FOLDER_NAME)

    
    def test():

        

        # train all the models
        GNN.eval()
        MLP2.eval()
        AlignmentClass.eval()
        SizeClass.eval()
        ElementGroupingClass.eval()
        HorizontalGroupingClass.eval()
        VerticalGroupingClass.eval()
        MultimodalGroupingClass.eval()
        PosEmbed.eval()
        TypeEmbed.eval()
        SizeEmbed.eval()

        # get target elements
        full_ui_path = './dataset/json_full_UIs/' + ui_name + '.json'

        with open(full_ui_path) as f:
            full_ui = json.load(f)

        target_elements = [ele for ele in full_ui['elements'] if ele not in orig_json_data['elements']]

        #orig_json_data = copy.deepcopy(json_data)

        result_target_list = []
        for target in target_elements:
            json_data = copy.deepcopy(orig_json_data)
            json_data['target'] = target

            json_data = add_constraints_to_partial_UI(copy.deepcopy(full_ui), copy.deepcopy(json_data))


            # get the data (all the pt files)
            data, ui_width_height, ele_text_embedding, ele_image_embedding, ele_type, target_text_embedding, target_image_embedding, \
                target_type, binary_target_alignment_links, binary_target_size_links, binary_target_element_grouping_links, \
                binary_target_horizontal_grouping_links, binary_target_vertical_grouping_links, binary_target_multimodal_grouping_links \
                = process(json_data=json_data, ui_name=ui_name)

            result_left = None
            result_right = None
            result_verticalmid = None
            result_top = None
            result_bottom = None
            result_horizontalmid = None
            result_width = None
            result_height = None

            result_left_candidates = []
            result_right_candidates = []
            result_verticalmid_candidates = []
            result_top_candidates = []
            result_bottom_candidates = []
            result_horizontalmid_candidates = []
            result_width_candidates = []
            result_height_candidates = []

            good_vertical_align = False
            good_horizontal_align = False
            good_size = False

            # if step == 500:
            #     exit()
            

            #file_basename, data, folder = item
            #batch_size = len(folder)
            # print(file_basename, folder)

            # if file_basename[0] != 'Rico_40834_381':
            #     continue
            # else:
            #     print('....')
                
      


            # get json file
            # with open(raw_orig_path + file_basename[0] + '.json') as f:
            #     json_data = json.load(f)
            id_to_ele_map = {} # key: ele id, value: index in the ele list
            for i in range(len(json_data['elements'])):
                ele = json_data['elements'][i]
                ele_id = ele['id']
                id_to_ele_map[ele_id] = ele
            target_ele_id = json_data['target']['id']
            target_aspect_ratio = json_data['target']['width'] / json_data['target']['height']

            element_size = data.x_dict['element'].shape[0]

            # initialization
            data = data.to(device)

            # get the node embeddings 
            # Note f is always 0 for testing
            # ele_text_embedding = torch.load(folder[f] + '/ele_text_embedding.pt').to(device)
            # ele_image_embedding = torch.load(folder[f] + '/ele_image_embedding.pt').to(device)
            # ele_type = torch.load(folder[f] + '/ele_type.pt').to(device)            
            ele_type_embedding = TypeEmbed(torch.argmax(ele_type, dim=1))
            ele_position_embedding = PosEmbed(data.x_dict['element'][:,:2])
            ele_size_embedding = SizeEmbed(data.x_dict['element'][:,2:])
            
            node_embedding = torch.concat((ele_position_embedding, ele_size_embedding, 
                                           ele_type_embedding, ele_image_embedding, 
                                           ele_text_embedding), axis=1)
            data.x_dict['element'] = node_embedding
            
            # get graph embedding, cell embeddings and constraint embeddings
            x_dict = data.x_dict
            graph_embedding, alignment_embeddings, size_embeddings, \
                 element_grouping_embeddings, horizontal_grouping_embeddings,\
                vertical_grouping_embeddings, multimodal_grouping_embeddings \
                                = GNN(x_dict, data.edge_index_dict, data, 1)

            # get target embedding        
            target_type_embedding = TypeEmbed(torch.argmax(target_type, dim=0)).unsqueeze(0)

            target_embedding = torch.concat((target_type_embedding, 
                                target_image_embedding.unsqueeze(0), target_text_embedding.unsqueeze(0)), axis=1)


            # predict final position and size
            pred_element = MLP2(torch.cat((graph_embedding, target_embedding), 1))

            # get predicted position and size
            pred_left = pred_element[0][0]
            pred_top = pred_element[0][1]
            pred_width = pred_element[0][2]
            pred_height = pred_element[0][3]
            pred_right = pred_left + pred_width
            pred_bottom = pred_top + pred_height
            pred_verticalmid = (pred_left + pred_right) * 0.5
            pred_horizontalmid = (pred_top + pred_bottom) * 0.5
            # print(pred_element, data['element'].y)

            # compute the loss for alignment classification for the target element node
            # i.e., whether the new element node has links to the alignments
            if alignment_embeddings != None:
                alignmentclass_input = torch.cat((torch.index_select(graph_embedding, 0, torch.LongTensor([0] * x_dict['alignment'].shape[0]))
                        , alignment_embeddings, torch.index_select(target_embedding, 0, torch.LongTensor([0] * x_dict['alignment'].shape[0]))), 1)
                pred_alignment = AlignmentClass(alignmentclass_input).reshape(-1)

                # get ground truth tags
                target_alignment = binary_target_alignment_links

                indices_alignment_target = (target_alignment == 1).nonzero(as_tuple=True)[0]
                indices_alignment_pred = (pred_alignment > 0).nonzero(as_tuple=True)[0]

            
                # get all the alignments
                for i in indices_alignment_pred:

                    constraint = data['alignment'].y[i]
                    if constraint[0] == 'left':
                        try:
                            ele_id_list = json_data['alignment'][constraint[0]][str(constraint[1])]
                        except:
                            ele_id_list = json_data['alignment'][constraint[0]][str(int(constraint[1]))]
                        try:
                            ele_id_list.remove(target_ele_id)
                        except:
                            pass
                        value_list = []
                        for ele_id in ele_id_list:
                            value_list.append(id_to_ele_map[ele_id]['left'])
                        value = sum(value_list) / len(value_list)
                        result_left_candidates.append(value)
                    if constraint[0] == 'right':
                        try:
                            ele_id_list = json_data['alignment'][constraint[0]][str(constraint[1])]
                        except:
                            ele_id_list = json_data['alignment'][constraint[0]][str(int(constraint[1]))]
                        try:
                            ele_id_list.remove(target_ele_id)
                        except:
                            pass
                        value_list = []
                        for ele_id in ele_id_list:
                            value_list.append(id_to_ele_map[ele_id]['right'])
                        value = sum(value_list) / len(value_list)
                        result_right_candidates.append(value)
                    if constraint[0] == 'vertical_midline':
                        try:
                            ele_id_list = json_data['alignment'][constraint[0]][str(constraint[1])]
                        except:
                            ele_id_list = json_data['alignment'][constraint[0]][str(int(constraint[1]))]
                        try:
                            ele_id_list.remove(target_ele_id)
                        except:
                            pass
                        value_list = []
                        for ele_id in ele_id_list:
                            value_list.append((id_to_ele_map[ele_id]['left'] + id_to_ele_map[ele_id]['right']) * 0.5)
                        value = sum(value_list) / len(value_list)
                        result_verticalmid_candidates.append(value)
                    if constraint[0] == 'top':
                        try:
                            ele_id_list = json_data['alignment'][constraint[0]][str(constraint[1])]
                        except:
                            ele_id_list = json_data['alignment'][constraint[0]][str(int(constraint[1]))]
                        try:
                            ele_id_list.remove(target_ele_id)
                        except:
                            pass
                        value_list = []
                        for ele_id in ele_id_list:
                            value_list.append(id_to_ele_map[ele_id]['top'])
                        value = sum(value_list) / len(value_list)
                        result_top_candidates.append(value)
                    if constraint[0] == 'bottom':
                        try:
                            ele_id_list = json_data['alignment'][constraint[0]][str(constraint[1])]
                        except:
                            ele_id_list = json_data['alignment'][constraint[0]][str(int(constraint[1]))]
                        try:
                            ele_id_list.remove(target_ele_id)
                        except:
                            pass
                        value_list = []
                        for ele_id in ele_id_list:
                            value_list.append(id_to_ele_map[ele_id]['bottom'])
                        value = sum(value_list) / len(value_list)
                        result_bottom_candidates.append(value)
                    if constraint[0] == 'horizontal_midline':
                        try:
                            ele_id_list = json_data['alignment'][constraint[0]][str(constraint[1])]
                        except:
                            ele_id_list = json_data['alignment'][constraint[0]][str(int(constraint[1]))]
                        try:
                            ele_id_list.remove(target_ele_id)
                        except:
                            pass
                        value_list = []
                        for ele_id in ele_id_list:
                            value_list.append((id_to_ele_map[ele_id]['top'] + id_to_ele_map[ele_id]['bottom']) * 0.5)
                        value = sum(value_list) / len(value_list)
                        result_horizontalmid_candidates.append(value)
                # print('\n========== Alignment ============')
                # print(indices_alignment_target)
                # print(indices_alignment_pred)
                # print('=================================\n')

                # get the final result based on alignments
                for value in result_left_candidates:
                    # move only if the element is close
                    if torch.abs(value - pred_left) <= max(100, pred_width): 
                        good_vertical_align = True
                        if result_left == None:
                            result_left = value
                        elif torch.abs(value - pred_left) < torch.abs(result_left - pred_left):
                            result_left = value

                for value in result_right_candidates:
                    # move only if the element is close
                    if torch.abs(value - pred_right) <= max(100, pred_width): 
                        good_vertical_align = True
                        if result_right == None:
                            result_right = value
                        elif torch.abs(value - pred_right) < torch.abs(result_right - pred_right):
                            result_right = value

                for value in result_verticalmid_candidates:
                    if torch.abs(value - pred_verticalmid) <= max(100, pred_width): 
                        good_vertical_align = True
                        if result_verticalmid == None:
                            result_verticalmid = value
                        elif torch.abs(value - pred_verticalmid) < torch.abs(result_verticalmid - pred_verticalmid):
                            result_verticalmid = value

                for value in result_top_candidates:
                    if torch.abs(value - pred_top) <= max(100, pred_height): 
                        good_horizontal_align = True
                        if result_top == None:
                            result_top = value
                        elif torch.abs(value - pred_top) < torch.abs(result_top - pred_top):
                            result_top = value

                for value in result_bottom_candidates:
                    if torch.abs(value - pred_bottom) <= max(100, pred_height): 
                        good_horizontal_align = True
                        if result_bottom == None:
                            result_bottom = value
                        elif torch.abs(value - pred_bottom) < torch.abs(result_bottom - pred_bottom):
                            result_bottom = value

                for value in result_horizontalmid_candidates:
                    if torch.abs(value - pred_horizontalmid) <= max(100, pred_height): 
                        good_horizontal_align = True
                        if result_horizontalmid == None:
                            result_horizontalmid = value
                        elif torch.abs(value - pred_horizontalmid) < torch.abs(result_horizontalmid - pred_horizontalmid):
                            result_horizontalmid = value


            # compute the loss for size classification for the target element node
            # i.e., whether the new element node has links to the sizes
            if size_embeddings != None:
                sizeclass_input = torch.cat((torch.index_select(graph_embedding, 0, torch.LongTensor([0] * x_dict['size'].shape[0]))
                        , size_embeddings, torch.index_select(target_embedding, 0, torch.LongTensor([0] * x_dict['size'].shape[0]))), 1)
                pred_size = SizeClass(sizeclass_input).reshape(-1)

                # get ground truth tags
                target_size = binary_target_size_links
                indices_size_target = (target_size == 1).nonzero(as_tuple=True)[0]
                indices_size_pred = (pred_size > 0).nonzero(as_tuple=True)[0]
                # print('\n============= Size ==============')
                # print(indices_size_target)
                # print(indices_size_pred)
                # print('=================================\n')


                # get all the size constraints
                for i in indices_size_pred:
                    constraint = data['size'].y[i]
                    if constraint[0] == 'width':
                        
                        try:
                            w_id_list = json_data['size'][constraint[0]][str(constraint[1])]
                        except:
                            w_id_list = json_data['size'][constraint[0]][str(int(constraint[1]))]

                        w_list = []
                        for w_id in w_id_list:
                            if w_id != target_ele_id:
                                w_list.append(id_to_ele_map[w_id]['width'])

                        result_width_candidates.append(sum(w_list) / len(w_list))
                    if constraint[0] == 'height':
                        
                        try:
                            h_id_list = json_data['size'][constraint[0]][str(constraint[1])]
                        except:
                            h_id_list = json_data['size'][constraint[0]][str(int(constraint[1]))]

                        h_list = []
                        for h_id in h_id_list:
                            if h_id != target_ele_id:
                                h_list.append(id_to_ele_map[h_id]['height'])

                        result_height_candidates.append(sum(h_list) / len(h_list))                    

                for value in result_width_candidates:
                    # if torch.abs(value - pred_width) <= 0.25 * pred_width: 
                    good_size = True
                    if result_width == None:
                        result_width = value
                    elif torch.abs(value - pred_width) < torch.abs(result_width - pred_width):
                        result_width = value

                for value in result_height_candidates:
                    # if torch.abs(value - pred_height) <= 0.25 * pred_height: 
                    good_size = True
                    if result_height == None:
                        result_height = value
                    elif torch.abs(value - pred_height) < torch.abs(result_height - pred_height):
                        result_height = value

                # if we have better combination, select the one better fitting the aspect ratio
                if result_width and result_height:
                    for value_w in result_width_candidates:
                        for value_h in result_height_candidates:
                            if abs(value_h * target_aspect_ratio - value_w) <= 10:
                                result_width = value_w
                                result_height = value_h

                # we use height as the more important dimension
                if result_height != None:
                    result_width = result_height * target_aspect_ratio
                elif result_height == None and result_width != None:
                    result_height = result_width / target_aspect_ratio

                # print('--', result_left, result_right, result_top, result_bottom, result_width, result_height)


                # if there is any identical element, we use that width and height
                width_id_list = []
                height_id_list = []
                for i in indices_size_pred:
                    constraint = data['size'].y[i]
                    if constraint[0] == 'width':

                        try:
                            width_id_list += json_data['size'][constraint[0]][str(constraint[1])]
                        except:
                            width_id_list += json_data['size'][constraint[0]][str(int(constraint[1]))]

                    if constraint[0] == 'height':
                        
                        try:
                            height_id_list += json_data['size'][constraint[0]][str(constraint[1])]
                        except:
                            height_id_list += json_data['size'][constraint[0]][str(int(constraint[1]))]
                        
                identical_ele_id_set = set(width_id_list).intersection(set(height_id_list))

                #identical_ele_id_set = list(identical_ele_id_set)
                try:
                    identical_ele_id_set.remove(target_ele_id)
                except:
                    pass
                if identical_ele_id_set != set():
                    closest_result_width = None
                    closest_result_height = None

                    for iden_ele_id in list(identical_ele_id_set):

                        if closest_result_width == None:
                            closest_result_width = id_to_ele_map[iden_ele_id]['width']
                            closest_result_height = id_to_ele_map[iden_ele_id]['height']
                        else:
                            curr_width = id_to_ele_map[iden_ele_id]['width']
                            curr_height = id_to_ele_map[iden_ele_id]['height']
                            if (abs(curr_width - result_width) * abs(curr_height - result_height)) \
                                < (abs(closest_result_width - result_width) * abs(closest_result_height - result_height)):
                                closest_result_width = curr_width
                                closest_result_height = curr_height
                    result_width = closest_result_width
                    result_height = closest_result_height



            if result_left != None and result_width != None:
                result_right = result_left + result_width
            if result_top != None and result_height != None:
                result_bottom = result_top + result_height


            if result_verticalmid != None:
                if result_width != None and result_left == None and result_right == None:
                    result_left = result_verticalmid - 0.5 * result_width
                    result_right = result_verticalmid + 0.5 * result_width

                if result_left == None and result_right != None and result_width == None:
                    # result_width = (result_right - result_verticalmid) * 2
                    result_left = 2 * result_verticalmid - result_right

                if result_left != None and result_right == None and result_width == None:
                    # result_width = (result_verticalmid - result_left) * 2
                    result_right = 2 * result_verticalmid - result_left

            if result_horizontalmid != None:
                if result_height != None and result_top == None and result_bottom == None:
                    result_top = result_horizontalmid - 0.5 * result_height
                    result_bottom = result_horizontalmid + 0.5 * result_height

                if result_top == None and result_bottom != None and result_height == None:
                    # result_width = (result_bottom - result_horizontalmid) * 2
                    result_top = 2 * result_horizontalmid - result_bottom

                if result_top != None and result_bottom == None and result_height == None:
                    # result_width = (result_horizontalmid - result_top) * 2
                    result_bottom = 2 * result_horizontalmid - result_top

            # compute position and size based on existing data
            # if result_left != None and result_right != None and result_width == None:
            #     result_width = result_right - result_left
            if result_left != None and result_right == None and result_width != None:
                result_right = result_left + result_width
            if result_left == None and result_right != None and result_width != None:
                result_left = result_right - result_width
            # if result_top != None and result_bottom != None and result_height == None:
            #     result_height = result_bottom - result_top
            if result_top != None and result_bottom == None and result_height != None:
                result_bottom = result_top + result_height
            if result_top == None and result_bottom != None and result_height != None:
                result_top = result_bottom - result_height



            

            if good_size and good_vertical_align and good_horizontal_align \
              and (abs(result_width - data['element'].y[0, 2].item()) < 15 \
                or abs(result_height - data['element'].y[0, 3].item()) < 15):
                json_data['target']['left'] = result_left
                json_data['target']['right'] = result_right
                json_data['target']['width'] = result_width
                json_data['target']['top'] = result_top
                json_data['target']['bottom'] = result_bottom
                json_data['target']['height'] = result_height

                try:

                    if check_overlap(json_data['target'], json_data['elements']):
                        json_data['target']['level'] = 'low'
                        result_target_list.append(json_data['target'])
                except:
                    json_data['target']['level'] = 'low'
                    result_target_list.append(json_data['target'])

                try:
                    json_data['target']['level'] = 'high'
                    result_target_list.append(json_data['target'])
                    # # Serializing json
                    # json_object = json.dumps(json_data, indent=4)
                     
                    # # Writing to sample.json
                    # with open(raw_result_path_high + file_basename[0] + '.json', "w") as outfile:
                    #     outfile.write(json_object)
                except:
                    pass
                json_data['target']['level'] = 'low'
                result_target_list.append(json_data['target'])

            # print('====', element_grouping_embeddings)
            # exit()



                        
            # compute the loss for element grouping classification for the target element node
            # i.e., whether the new element node has links to the element grouping 
            if element_grouping_embeddings != None:
                element_groupingclass_input = torch.cat((torch.index_select(graph_embedding, 0, torch.LongTensor([0] * x_dict['element_grouping'].shape[0]))
                        , element_grouping_embeddings, torch.index_select(target_embedding, 0, torch.LongTensor([0] * x_dict['element_grouping'].shape[0]))), 1)
                pred_element_grouping = ElementGroupingClass(element_groupingclass_input).reshape(-1)

                # get ground truth tags
                target_element_grouping = binary_target_element_grouping_links

                # get the indices of the groupings
                indices_element_grouping_target = (target_element_grouping == 1).nonzero(as_tuple=True)[0]
                indices_element_grouping_pred = (pred_element_grouping > 0).nonzero(as_tuple=True)[0]
                # print('\n======= Element Grouping ========')
                # print(indices_element_grouping_target)
                # print(indices_element_grouping_pred)
                # print('=================================\n')

                # check vertial element grouping
                w_list = []
                h_list = []
                d_list = []
                for i in indices_element_grouping_pred:
                    constraint = data['element_grouping'].y[i]
                    if constraint[0] == 'vertical':
                        w_list.append(constraint[1])
                        h_list.append(constraint[2])
                        d_list.append(constraint[3])

                # when we have the vertical element grouping
                # we get the closest vertical element grouping
                if w_list != []:

                    # find the most possible element grouping
                    best_group_id = 0
                    w_h_dist_multiply = abs(w_list[0] - pred_width) * abs(h_list[0] - pred_height)
                    for i in range(1, len(w_list)):
                        w_h_dist_multiply_new = abs(w_list[i] - pred_width) * abs(h_list[i] - pred_height)
                        if w_h_dist_multiply_new < w_h_dist_multiply:
                            w_h_dist_multiply = w_h_dist_multiply_new
                            best_group_id = i

                    w = w_list[best_group_id]
                    h = h_list[best_group_id]
                    d = d_list[best_group_id]

                    grouping_str = str(constraint[1]) + '#' + str(constraint[2]) + '#' + str(constraint[3])
                    ele_id_list = json_data['element grouping'][constraint[0]][grouping_str]
                    
                    # get all the vertical midline in the group
                    left_list = []
                    height_list = []
                    width_list = []
                    ele_id_list_final = []
                    v_mid_list = []
                    d_list = []
                    for idx in range(len(ele_id_list)):
                        ele_id = ele_id_list[idx]

                        # remove target
                        if ele_id in id_to_ele_map.keys():
                            ele = id_to_ele_map[ele_id]
                            left_list.append(ele['left'])
                            v_mid_list.append((ele['left'] + ele['right']) * 0.5)
                            height_list.append(ele['height'])
                            width_list.append(ele['width'])
                            ele_id_list_final.append(ele_id)

                    for idx in range(1, len(ele_id_list_final)):
                        ele_id = ele_id_list_final[idx]
                        ele_id_prev = ele_id_list_final[idx - 1]
                        ele = id_to_ele_map[ele_id]
                        ele_prev = id_to_ele_map[ele_id_prev]
                        d_list.append(ele['top'] - ele_prev['top'])

                    try:
                        d = sum(d_list) / len(d_list)
                    except:
                        pass

                    left = sum(left_list) / len(left_list)
                    height = sum(height_list) / len(height_list)
                    width = sum(width_list) / len(width_list)
                    if max(v_mid_list) - min(v_mid_list) < 10:
                        v_mid = sum(v_mid_list) / len(v_mid_list)
                    else: 
                        v_mid = None

                    # avoid the distance variation to be too large
                    if d_list != [] and max(d_list) - min(d_list) < min(d_list):
                        if result_width == None and result_height == None:
                            if abs(height - pred_height) <= abs(width - pred_width):
                                result_height = height
                                result_width = result_height * target_aspect_ratio
                            else:
                                result_width = width
                                result_height = result_width / target_aspect_ratio
                
                        if v_mid != None:
                            if result_left == None:
                                result_left = v_mid - 0.5 * result_width
                            if result_right == None:
                                result_right = v_mid + 0.5 * result_width
                        else:
                            if result_left == None:
                                result_left = left
                            if result_right == None:
                                result_right = result_left + result_width
                        # print(result_height, result_width)

                        if result_top == None and result_bottom == None:
                            # decide to place at the top or bottom
                            if abs(pred_top - id_to_ele_map[ele_id_list_final[-1]]['bottom']) \
                                < abs(pred_bottom - id_to_ele_map[ele_id_list_final[0]]['top']) + 500:
                                result_top = id_to_ele_map[ele_id_list_final[-1]]['top'] + d
                                result_bottom = result_top + result_height
                            else:
                                result_bottom = id_to_ele_map[ele_id_list_final[0]]['bottom'] - d
                                result_top = result_bottom - result_height

                        good_vertical_align = True
                        good_horizontal_align = True
                        good_size = True

                # check horizontal element grouping
                w_list = []
                h_list = []
                d_list = []
                for i in indices_element_grouping_pred:

                    constraint = data['element_grouping'].y[i]
                    if constraint[0] == 'horizontal':
                        w_list.append(constraint[1])
                        h_list.append(constraint[2])
                        d_list.append(constraint[3])

                # when we have the vertical element grouping
                # we get the closest vertical element grouping
                if w_list != []:

                    # find the most possible element grouping
                    best_group_id = 0
                    w_h_dist_multiply = abs(w_list[0] - pred_width) * abs(h_list[0] - pred_height)
                    for i in range(1, len(w_list)):
                        w_h_dist_multiply_new = abs(w_list[i] - pred_width) * abs(h_list[i] - pred_height)
                        if w_h_dist_multiply_new < w_h_dist_multiply:
                            w_h_dist_multiply = w_h_dist_multiply_new
                            best_group_id = i

                    w = w_list[best_group_id]
                    h = h_list[best_group_id]
                    d = d_list[best_group_id]
                    

                    grouping_str = str(constraint[1]) + '#' + str(constraint[2]) + '#' + str(constraint[3])
                    ele_id_list = json_data['element grouping'][constraint[0]][grouping_str]
                    
                    # get all the vertical midline in the group
                    top_list = []
                    width_list = []
                    height_list = []
                    ele_id_list_final = []
                    h_mid_list = []
                    d_list = []
                    for ele_id in ele_id_list:

                        # remove target
                        if ele_id in id_to_ele_map.keys():
                            ele = id_to_ele_map[ele_id]
                            top_list.append(ele['top'])
                            h_mid_list.append((ele['top'] + ele['bottom']) * 0.5)
                            width_list.append(ele['width'])
                            height_list.append(ele['height'])
                            ele_id_list_final.append(ele_id)

                    for idx in range(1, len(ele_id_list_final)):
                        ele_id = ele_id_list_final[idx]
                        ele_id_prev = ele_id_list_final[idx - 1]
                        ele = id_to_ele_map[ele_id]
                        ele_prev = id_to_ele_map[ele_id_prev]
                        d_list.append(ele['left'] - ele_prev['left'])

                    try:
                        d = sum(d_list) / len(d_list)
                    except:
                        pass 

                    top = sum(top_list) / len(top_list)
                    width = sum(width_list) / len(width_list)
                    height = sum(height_list) / len(height_list)
                    if max(h_mid_list) - min(h_mid_list) < 10:
                        h_mid = sum(h_mid_list) / len(h_mid_list)
                    else: 
                        h_mid = None

                    if d_list != [] and max(d_list) - min(d_list) < min(d_list):
                        if result_width == None and result_height == None:
                            if abs(height - pred_height) <= abs(width - pred_width):
                                result_height = height
                                result_width = result_height * target_aspect_ratio
                            else:
                                result_width = width
                                result_height = result_width / target_aspect_ratio

                        if h_mid != None:
                            if result_top == None:
                                result_top = h_mid - 0.5 * result_height
                            if result_bottom == None:
                                result_bottom = h_mid + 0.5 * result_height
                        else:
                            if result_top == None:
                                result_top = top
                            if result_bottom == None:
                                result_bottom = result_top + result_height

                        if result_left == None and result_right == None:
                            # decide to place at the top or bottom
                            if abs(pred_left - id_to_ele_map[ele_id_list_final[-1]]['right']) \
                                < abs(pred_right - id_to_ele_map[ele_id_list_final[0]]['left']) + 500:
                                result_left = id_to_ele_map[ele_id_list_final[-1]]['left'] + d
                                result_right = result_left + result_width
                            else:
                                result_right = id_to_ele_map[ele_id_list_final[0]]['right'] - d
                                result_left = result_right - result_width

                        good_vertical_align = True
                        good_horizontal_align = True
                        good_size = True


            ################################
            ###### Vertical Grouping #####
            ################################
            # compute the loss for element grouping classification for the target element node
            # i.e., whether the new element node has links to the element grouping 
            if vertical_grouping_embeddings != None:
                vertical_groupingclass_input = torch.cat((torch.index_select(graph_embedding, 0, torch.LongTensor([0] * x_dict['vertical_grouping'].shape[0]))
                        , vertical_grouping_embeddings, torch.index_select(target_embedding, 0, torch.LongTensor([0] * x_dict['vertical_grouping'].shape[0]))), 1)
                pred_vertical_grouping = VerticalGroupingClass(vertical_groupingclass_input).reshape(-1)

                # get ground truth tags
                target_vertical_grouping = binary_target_vertical_grouping_links

                # get the indices of the groupings
                indices_vertical_grouping_target = (target_vertical_grouping == 1).nonzero(as_tuple=True)[0]
                indices_vertical_grouping_pred = (pred_vertical_grouping > 0).nonzero(as_tuple=True)[0]

                # check vertial element grouping
                l_list = []
                r_list = []
                w_list = []
                h_list = []
                for i in indices_vertical_grouping_pred:
                    constraint = data['vertical_grouping'].y[i]
                    if constraint[0] == 'vertical':
                        l_list.append(constraint[1])
                        r_list.append(constraint[2])
                        w_list.append(constraint[3])
                        h_list.append(constraint[4])


                # when we have the vertical element grouping
                # we get the closest vertical element grouping
                if w_list != []:

                    # find the most possible element grouping
                    best_group_id = 0
                    w_h_dist_multiply = abs(w_list[0] - pred_width) * abs(h_list[0] - pred_height)
                    for i in range(1, len(w_list)):
                        w_h_dist_multiply_new = abs(w_list[i] - pred_width) * abs(h_list[i] - pred_height)
                        if w_h_dist_multiply_new < w_h_dist_multiply:
                            w_h_dist_multiply = w_h_dist_multiply_new
                            best_group_id = i

                    w = w_list[best_group_id]
                    h = h_list[best_group_id]
                    l = l_list[best_group_id]
                    r = r_list[best_group_id]

                    grouping_str = str(int(constraint[1])) + '/' + str(int(constraint[2]))
                    ele_id_list = json_data['vertical_groups'][grouping_str]

                    
                    result_left = l
                    result_right = r
                    result_width = w
                    result_height = h
                
                
                    good_vertical_align = True
                    # good_horizontal_align = True
                    good_size = True


            ################################
            ###### Horizontal Grouping #####
            ################################
            # compute the loss for element grouping classification for the target element node
            # i.e., whether the new element node has links to the element grouping 
            if horizontal_grouping_embeddings != None:
                horizontal_groupingclass_input = torch.cat((torch.index_select(graph_embedding, 0, torch.LongTensor([0] * x_dict['horizontal_grouping'].shape[0]))
                        , horizontal_grouping_embeddings, torch.index_select(target_embedding, 0, torch.LongTensor([0] * x_dict['horizontal_grouping'].shape[0]))), 1)
                pred_horizontal_grouping = HorizontalGroupingClass(horizontal_groupingclass_input).reshape(-1)

                # get ground truth tags
                target_horizontal_grouping = binary_target_horizontal_grouping_links

                # get the indices of the groupings
                indices_horizontal_grouping_target = (target_horizontal_grouping == 1).nonzero(as_tuple=True)[0]
                indices_horizontal_grouping_pred = (pred_horizontal_grouping > 0).nonzero(as_tuple=True)[0]

                # check vertial element grouping
                t_list = []
                b_list = []
                w_list = []
                h_list = []
                for i in indices_horizontal_grouping_pred:
                    constraint = data['horizontal_grouping'].y[i]
                    if constraint[0] == 'horizontal':
                        t_list.append(constraint[1])
                        b_list.append(constraint[2])
                        w_list.append(constraint[3])
                        h_list.append(constraint[4])


                # when we have the horizontal element grouping
                # we get the closest horizontal element grouping
                if w_list != []:

                    # find the most possible element grouping
                    best_group_id = 0
                    w_h_dist_multiply = abs(w_list[0] - pred_width) * abs(h_list[0] - pred_height)
                    for i in range(1, len(w_list)):
                        w_h_dist_multiply_new = abs(w_list[i] - pred_width) * abs(h_list[i] - pred_height)
                        if w_h_dist_multiply_new < w_h_dist_multiply:
                            w_h_dist_multiply = w_h_dist_multiply_new
                            best_group_id = i

                    w = w_list[best_group_id]
                    h = h_list[best_group_id]
                    t = t_list[best_group_id]
                    b = b_list[best_group_id]

                    grouping_str = str(int(constraint[1])) + '/' + str(int(constraint[2]))
                    ele_id_list = json_data['horizontal_groups'][grouping_str]

                    
                    result_top = t
                    result_bottom = b
                    result_width = w
                    result_height = h
                
                
                    # good_vertical_align = True
                    good_horizontal_align = True
                    good_size = True



            # print(json_data['target']['width'] - data['element'].y[0, 2].item())
            # print(json_data['target']['width'], data['element'].y[0, 2].item())
            # print(abs(json_data['target']['width'] - data['element'].y[0, 2].item()))
            # print(abs(json_data['target']['width'] - data['element'].y[0, 2].item()) > 10)
            # if the size is not roughly correct, then we do not save it
            # if abs(result_width - data['element'].y[0, 2].item()) > 10 \
            #     or abs(result_height - data['element'].y[0, 3].item()) > 10:
            #     good_size = False


            if good_size and good_vertical_align and good_horizontal_align \
              and (abs(result_width - data['element'].y[0, 2].item()) < 15 \
                and abs(result_height - data['element'].y[0, 3].item()) < 15):
                json_data['target']['left'] = result_left
                json_data['target']['right'] = result_right
                json_data['target']['width'] = result_width
                json_data['target']['top'] = result_top
                json_data['target']['bottom'] = result_bottom
                json_data['target']['height'] = result_height

                try:
                    if check_overlap(json_data['target'], json_data['elements']):
                        json_data['target']['level'] = 'low'
                        result_target_list.append(json_data['target'])
                except:
                    json_data['target']['level'] = 'low'
                    result_target_list.append(json_data['target'])

                try:
                    json_data['target']['level'] = 'medium'
                    result_target_list.append(json_data['target'])
                    # # Serializing json
                    # json_object = json.dumps(json_data, indent=4)
                    # print('-==-', result_left, result_right, result_top, result_bottom)
                     
                    # # Writing to sample.json
                    # with open(raw_result_path_median + file_basename[0] + '.json', "w") as outfile:
                    #     outfile.write(json_object)
                except:
                    pass
                json_data['target']['level'] = 'low'
                result_target_list.append(json_data['target'])

            # compute the loss for multimodal grouping classification for the target element node
            # i.e., whether the new element node has links to the multimodal grouping 
            if multimodal_grouping_embeddings != None:
                multimodal_groupingclass_input = torch.cat((torch.index_select(graph_embedding, 0, torch.LongTensor([0] * x_dict['multimodal_grouping'].shape[0]))
                        , multimodal_grouping_embeddings, torch.index_select(target_embedding, 0, torch.LongTensor([0] * x_dict['multimodal_grouping'].shape[0]))), 1)
                pred_multimodal_grouping = MultimodalGroupingClass(multimodal_groupingclass_input).reshape(-1)

                # get ground truth tags
                target_multimodal_grouping = binary_target_multimodal_grouping_links


            # compute position and size based on existing data
            if result_verticalmid != None:
                if result_width != None:
                    result_left = result_verticalmid - 0.5 * result_width
                    result_right = result_verticalmid + 0.5 * result_width

                if result_left == None and result_right != None and result_width == None:
                    result_width = (result_right - result_verticalmid) * 2
                    result_left = result_verticalmid - 0.5 * result_width

                if result_left != None and result_right == None and result_width == None:
                    result_width = (result_verticalmid - result_left) * 2
                    result_right = result_verticalmid + 0.5 * result_width

            if result_horizontalmid != None:
                if result_height != None:
                    result_top = result_horizontalmid - 0.5 * result_height
                    result_bottom = result_horizontalmid + 0.5 * result_height

                if result_top == None and result_bottom != None and result_height == None:
                    result_height = (result_bottom - result_horizontalmid) * 2
                    result_top = result_horizontalmid - 0.5 * result_height

                if result_top != None and result_bottom == None and result_height == None:
                    result_height = (result_horizontalmid - result_top) * 2
                    result_bottom = result_horizontalmid + 0.5 * result_height

            if result_left != None and result_right != None and result_width == None:
                result_width = result_right - result_left
            if result_left != None and result_right == None and result_width != None:
                result_right = result_left + result_width
            if result_left == None and result_right != None and result_width != None:
                result_left = result_right - result_width
            if result_top != None and result_bottom != None and result_height == None:
                result_height = result_bottom - result_top
            if result_top != None and result_bottom == None and result_height != None:
                result_bottom = result_top + result_height
            if result_top == None and result_bottom != None and result_height != None:
                result_top = result_bottom - result_height

            if result_width == None and result_height != None:
                result_width = result_height * target_aspect_ratio
            elif result_width != None and result_height == None:
                result_height = result_width / target_aspect_ratio

            if result_width == None and result_height == None:
                result_height = pred_height
                result_width = result_height * target_aspect_ratio

            if abs(result_height - pred_height) <= abs(result_width - pred_width):
                result_width = result_height * target_aspect_ratio
            else:
                result_height = result_width / target_aspect_ratio

            if result_left == None:
                result_left = pred_left
            result_right = result_left + result_width
            if result_top == None:
                result_top = pred_top
            result_bottom = result_top + result_height

            if type(result_left) == float or type(result_left) == int:
                json_data['target']['left'] = result_left
            else:
                json_data['target']['left'] = result_left.item()

            if type(result_right) == float or type(result_right) == int:
                json_data['target']['right'] = result_right
            else:
                json_data['target']['right'] = result_right.item()

            if type(result_top) == float or type(result_top) == int:
                json_data['target']['top'] = result_top
            else:
                json_data['target']['top'] = result_top.item()

            if type(result_bottom) == float or type(result_bottom) == int:
                json_data['target']['bottom'] = result_bottom
            else:
                json_data['target']['bottom'] = result_bottom.item()

            if type(result_width) == float or type(result_width) == int:
                json_data['target']['width'] = result_width
            else:
                json_data['target']['width'] = result_width.item()

            if type(result_height) == float or type(result_height) == int:
                json_data['target']['height'] = result_height
            else:
                json_data['target']['height'] = result_height.item()
            if check_overlap(json_data['target'], json_data['elements']):
                json_data['target']['level'] = 'low'
                result_target_list.append(json_data['target'])

            json_data['target']['level'] = 'low'
            result_target_list.append(json_data['target'])

        orig_json_data['target'] = result_target_list    
         
        return orig_json_data

         

    return test()