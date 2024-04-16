import cv2 as cv
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

def resize_and_show(name, image, ax):
    ax.imshow(image)
    ax.set_title(name)
    # print(type(image))
    # h, w = image.shape[:2]
    # if h < w:
    #     img = cv.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    # else:
    #     img = cv.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    # cv.imshow(name, img)
    # cv.waitKey(0) 
    # cv.destroyAllWindows() 

def adjacent(image, segment, branches, branch_squares):


    return adj_nodes        #list of start and ending point of graph
    # nh = 2
    # adj_index = []
    # # start = segment[0]
    # # end = segment[-1]
    # ends = find_ends(segment)
    # start = ends[0]
    # end = ends[1]
    # index = 0
    # for node in nodes:
    #     if((abs(start[0]-node[0]) <= nh and abs(start[1]-node[1]) <= nh) or (abs(end[0]-node[0]) <= nh and abs(end[1]-node[1]) <= nh)):
    #         adj_index.append(index)
    #     index = index + 1
    # if(len(adj_index) != 2):
    #     print("error, segment:", start, end, "is adjacent to ", len(adj_index), " nodes")
    # return adj_index

def extract_square(image, coordinate):
    center_row = coordinate[0]
    center_col = coordinate[1]
    print(type(image))
    print(np.shape(image))
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    start_row = max(center_row - 1, 0)
    end_row = min(center_row + 1, height - 1)
    start_col = max(center_col - 1, 0)
    end_col = min(center_col + 1, width - 1)
    mask_image = np.zeros((height, width))
    #square = image[start_row:(end_row+1), start_col:(end_col+1)]
    mask_image[start_row:(end_row+1), start_col:(end_col+1)] = image[start_row:(end_row+1), start_col:(end_col+1)]
    
    # plt.imshow(mask_image)
    # plt.show()
    return mask_image

def get_neighbors(image, coord):
    row, col = coord
    intensities = set()
    neighbors = []
    num_rows = np.shape(image)[0]
    num_cols = np.shape(image)[1]

  # Check top neighbor
    if row > 0:
        intensities.add(image[row - 1, col])
        if(image[row, col] == image[row - 1, col]):
            neighbors.append([row - 1, col])
    # Check bottom neighbor
    if row < num_rows - 1:
        intensities.add(image[row + 1, col])
        if(image[row, col] == image[row + 1, col]):
            neighbors.append([row + 1, col])
    # Check left neighbor
    if col > 0:
        intensities.add(image[row, col - 1])
        if(image[row, col] == image[row, col - 1]):
            neighbors.append([row, col - 1])
    # Check right neighbor
    if col < num_cols - 1:
        intensities.add(image[row, col + 1])
        if(image[row, col] == image[row, col + 1]):
            neighbors.append([row, col + 1])
    # Check top-left neighbor
    if row > 0 and col > 0:
        intensities.add(image[row - 1, col - 1])
        if(image[row, col] == image[row - 1, col - 1]):
            neighbors.append([row - 1, col - 1])
    # Check top-right neighbor
    if row > 0 and col < num_cols - 1:
        intensities.add(image[row - 1, col + 1])
        if(image[row, col] == image[row - 1, col + 1]):
            neighbors.append([row - 1, col + 1])
    # Check bottom-left neighbor
    if row < num_rows - 1 and col > 0:
        intensities.add(image[row + 1, col - 1])
        if(image[row, col] == image[row + 1, col - 1]):
            neighbors.append([row + 1, col - 1])
    # Check bottom-right neighbor
    if row < num_rows - 1 and col < num_cols - 1:
        intensities.add(image[row + 1, col + 1])
        if(image[row, col] == image[row + 1, col + 1]):
            neighbors.append([row + 1, col + 1])

    return neighbors, intensities

def longest_path(G, root):
    if(G.adj[root] == None):
        return(root)

    max_path = None
    max_length = 0
    for adjacent in G.adj[root]:
        edge = G.get_edge_data(root, adjacent)
        length = edge["weight"]
        new_path, new_length = DFS(G, [root, adjacent], length)
        if(new_length > max_length):
            print("here")
            max_length = new_length
            max_path = list(new_path)
    return max_path

def DFS(G, path, length):
    root = path[-1]
    adjacent_nodes = G.adj[root]
    print("path: ", path, "     adjacent:", adjacent_nodes)
    if(all(node_id in path for node_id in list(adjacent_nodes.keys()))):            #check if adjacency list of the previous node is completly contained within path
        return(path, length)
    max_length = 0
    max_path = None
    for adjacent in adjacent_nodes:
        if(adjacent in path):
            continue
        edge = G.get_edge_data(root, adjacent)
        new_path = list(path)
        new_path.append(adjacent)
        print("new_path", new_path)
        new_length = length + edge["weight"]
        print(new_length)
        new_path, new_length = DFS(G, new_path, new_length)
        print("Next Length", new_length)
        if(new_length > max_length):    
            max_length = new_length
            max_path = new_path
            
    return max_path, max_length

def any_path(G, root, target):
    if(G.adj[root] == None):
        return(root)

    for adjacent in G.adj[root]:
        edge = G.get_edge_data(root, adjacent)
        new_path = path_rec(G, [root, adjacent], target)
        if(new_path != None and target in new_path):
            return(new_path)
    return None

def path_rec(G, path, target):
    root = path[-1]
    adjacent_nodes = G.adj[root]
    print("path: ", path, "     adjacent:", adjacent_nodes)
    if(all(node_id in path for node_id in list(adjacent_nodes.keys()))):            #check if adjacency list of the previous node is completly contained within path
        return(path)
    max_path = None
    for adjacent in adjacent_nodes:
        if(adjacent in path):
            continue
        new_path = list(path)
        new_path.append(adjacent)
        print("new_path", new_path)
        new_path = path_rec(G, new_path, target)
        if(new_path != None and target in new_path):    
            return(new_path)
            
    return None
        

def main():
    images = os.listdir("./images/")

    for file_name in images:

        image_file_name = "./images/" + file_name

        figure, (ax0, ax1, ax2) = plt.subplots(1, 3)
        figure.tight_layout()

        img = cv.imread(image_file_name)
        resize_and_show(image_file_name, img, ax0)
    
        base_options = python.BaseOptions(model_asset_path="./model/pose_landmarker.task", delegate=mp.tasks.BaseOptions.Delegate.CPU)
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE, output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)
        image = mp.Image.create_from_file(image_file_name)
        # image = cv.imread(image_file_name)
        # image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = detector.detect(image)

        # print(detection_result)
        segmentation_mask = 0 
        try:
            segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        except:
            print("No Human Detected")
            plt.show()
            continue


        print(segmentation_mask.dtype)
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2)
        from skimage import color
        visualized_mask = color.rgb2gray(visualized_mask)

        ax1.imshow(visualized_mask, cmap="gray")
        ax1.set_title("visualized mask")
        # resize_and_show(image_file_name, visualized_mask, ax1)
 
        from skimage.morphology import skeletonize, thin
        threshold = 0.5
        boolean_mask = visualized_mask >= threshold
        cv_skeleton = skeletonize(boolean_mask)

        ax2.imshow(cv_skeleton, cmap=plt.cm.gray)
        ax2.set_title("skeleton")

        plt.show()

        from plantcv import plantcv as pcv
        from skimage import measure
        from skimage.morphology import binary_dilation
        import skimage
        import networkx as nx
        # pcv.params.debug = "plot"


        ##extract largest connected component from image##
        single_component = boolean_mask * 0
        labels, num_labels = measure.label(boolean_mask, connectivity=2, return_num = True)  # 8-connectivity
        regions = measure.regionprops(labels)
        print("num_regions", len(regions))
        reg_ind = [i+1 for i in range(len(regions))]
        print(num_labels)
        component_sizes = [region.area for region in regions]
        component_sizes, reg_ind = (list(t) for t in zip(*sorted(zip(component_sizes, reg_ind))))
        print(component_sizes)
        print("largest component: ", reg_ind[-1])
        comp_x, comp_y = np.where(labels == reg_ind[-1])
        component = np.column_stack((comp_x, comp_y))
        print(component)
        for pixel in component:
            single_component[pixel[0], pixel[1]] = 1
        plt.imshow(single_component)
        plt.title("single_component")
        # plt.show()
        
        ##remove holes by flooding region that is not connected to [0,0]##
        labels, num_labels = measure.label(single_component + 1, connectivity=2, return_num = True)
        regions = measure.regionprops(labels)
        component_coordinates = []
        background = None
        for region in regions:
            coordinates = [[coord[0], coord[1]] for coord in region.coords] 
            # Append coordinates to the list
            component_coordinates.append(coordinates)
        component = []
        for component in component_coordinates:
            if([0,0] in component):
                background = component 
        hole_mask = boolean_mask*0+1
        for pixel in background:
            hole_mask[pixel[0], pixel[1]] = 0

        plt.imshow(hole_mask)
        plt.title("filled_component")
        #plt.show()

        skeleton = pcv.morphology.skeletonize(mask=hole_mask)                    #plantcv skeleton object     
        plt.imshow(skeleton)
        plt.title("skeleton")
        #plt.show()

        branch_img = pcv.morphology.find_branch_pts(skel_img=skeleton)       #binary mask of branch points
        tip_img = pcv.morphology.find_tips(skel_img=skeleton)                      #binary mask of tip points
       
 
        pcv.params.line_thickness = 2                                              #increase line width for debug plots

        branch_row_indices, branch_col_indices = np.where(branch_img == 255)
        tip_row_indices, tip_col_indices = np.where(tip_img == 255)

             # Combine row and column indices to get coordinates
        branch_coordinates = np.column_stack((branch_row_indices, branch_col_indices))
        tip_coordinates = np.column_stack((tip_row_indices, tip_col_indices))
        for i in range(len(tip_coordinates)):
            neighbors = get_neighbors(skeleton, tip_coordinates[i])[0]
            if(len(neighbors) != 1):
                print("removing tip with {} neighbors".format(len(neighbors)))

        i=0
        segment_image_add = []            ##removed branch points are added to semgmented image
        while(i < len(branch_coordinates)):
            neighbors = get_neighbors(skeleton, branch_coordinates[i])[0]
            if(len(neighbors) <= 2):
                print("removing branch with {} neighbors".format(len(neighbors)))
                branch_coordinates = branch_coordinates[:i] + branch_coordinates[i+1:]
            for branch in branch_coordinates:
                for neighbor in neighbors:
                    if(neighbor[0] == branch[0] and neighbor[1] == branch[1]):
                        print("removing branch with {} neighbors".format(len(neighbors)))
                        branch_coordinates = np.delete(branch_coordinates, i, 0)
                        print("new_branch coordnates:\n", branch_coordinates, '\n', '\n')
                        segment_image_add.append([branch_coordiantes[i], i])
            i = i + 1
        
        cross_kernel = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]], dtype = np.uint8)
        dil_branch_img = branch_img
        dil_branch_img = cv.dilate(branch_img, cross_kernel)
        print("dialation min, max: ", np.min(dil_branch_img), np.max(dil_branch_img))
        #plt.imshow(dil_branch_img)
        #plt.show()
        segment_mask = np.clip(skeleton - dil_branch_img - tip_img, a_min = 2, a_max = 255) - 2  ## binary image of 0 and 253 intensities
        # segment_mask  = segment_mask + branch_img*0.5

     
        plt.imshow(segment_mask)
        plt.title("segment_mask")
        #plt.show()

        labels, num_labels = measure.label(segment_mask, connectivity=2, return_num = True)  # 8-connectivity
        print(np.max(labels))
        print(num_labels)
        figure = plt.imshow(labels)
        #plt.show()

        # Get region properties for each labeled region
        regions = measure.regionprops(labels)
        #regions.pop(0)          # remove background region
        # Extract coordinates for each region
        component_coordinates = []
        segment_sizes = []
        for region in regions:
            segment_sizes.append(region.area)
            # Extract coordinates for the current region
            coordinates = [[coord[0], coord[1]] for coord in region.coords] 
            # Append coordinates to the list
            component_coordinates.append(coordinates)

        print("length of segment_sizes: ", len(segment_sizes))

    
        print("num branches : ", branch_img.sum()/255)
        print("num tips : ", tip_img.sum()/255)
        print("branch_shape: ", np.shape(branch_coordinates))
        print("tip_shape: ", np.shape(tip_coordinates))
        print("branches: "+'\n', branch_coordinates)
        print("tips: "+'\n', tip_coordinates)

        G = nx.Graph()
        connectivity_mask = labels
        tip_label_start = num_labels +1
        print("tip_label_start: ", tip_label_start)
        tip_neighbors = []
        for i in range(len(tip_coordinates)):
            curr_label = tip_label_start + i
            G.add_node('t{}'.format(curr_label))
            coord = tip_coordinates[i][0], tip_coordinates[i][1]
            connectivity_mask[coord] = curr_label
            pixel_points, neighbors = get_neighbors(connectivity_mask, coord)  
            neighbors.discard(0)
            neighbors.discard(curr_label)
            print("tip neighbors: ", neighbors)
            tip_neighbors.append(neighbors)
        
        ##generate branch square masks
        branch_squares = [extract_square(skeleton, n) for n in branch_coordinates]
        branch_label_start = tip_label_start + len(tip_coordinates)
        print("branch_label_start: ", branch_label_start)
        ##add branch sqaure masks to connectivity mask, generate coordiantes of branch squares
        branch_neighbors = []
        for i in range(len(branch_coordinates)):
            curr_label = branch_label_start + i                          #label of current branch
            G.add_node('b{}'.format(curr_label))
            branch_mask = np.clip(branch_squares[i], a_min = None, a_max = 1) * curr_label
            connectivity_mask = connectivity_mask + branch_mask
            branch_row_indices, branch_col_indices = np.where(connectivity_mask == curr_label)
            branch_region = np.column_stack((branch_row_indices, branch_col_indices))
            ##check for connectivity##
            neighbors = set()
            for coord in branch_region:
                neighbors = neighbors | get_neighbors(connectivity_mask, coord)[1]  
            neighbors.discard(0)
            neighbors.discard(curr_label)
            print("branch neighbors: ", neighbors)
            branch_neighbors.append(neighbors)

        plt.imshow(connectivity_mask)
        plt.title("connectivity mask")
        #plt.show()

        for i in range(len(tip_neighbors)):
            start_set = tip_neighbors[i]
            for start in start_set:
                for n in range(len(tip_neighbors)):
                    end_set = tip_neighbors[n]
                    for end in end_set:
                        if(start == end and i != n):
                            G.add_edge("t{}".format(i+tip_label_start), "t{}".format(n+tip_label_start), weight = segment_sizes[int(start)-1], label = start)
                for n in range(len(branch_neighbors)):
                        end_set = branch_neighbors[n]
                        for end in end_set:
                            if(start == end):
                                G.add_edge("t{}".format(i+tip_label_start), "b{}".format(n+branch_label_start), weight = segment_sizes[int(start)-1], label = start)
        
        for i in range(len(branch_neighbors)):
            start_set = branch_neighbors[i]
            for start in start_set:
                for n in range(len(branch_neighbors)):
                    end_set = branch_neighbors[n]
                    for end in end_set:
                        if(start == end and i != n):
                            G.add_edge("b{}".format(i+branch_label_start), "b{}".format(n+branch_label_start), weight = segment_sizes[int(start)-1], label = start)
            


        # pos = nx.kamada_kawai_layout(G)
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold')
        nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
        #plt.show()

        root = next(iter(G.nodes()))
        root = "t{}".format(tip_label_start) 
        w = longest_path(G, root)[-1]
        path = longest_path(G, w)

        highest_ind = 0
        lowest_ind = 0
        for i in range(len(tip_coordinates)):
            print(tip_coordinates[i][0], ", ",tip_coordinates[highest_ind][0])
            if(tip_coordinates[i][0] > tip_coordinates[highest_ind][0]):
                highest_ind = i
            if(tip_coordinates[i][0] < tip_coordinates[highest_ind][0]):
                lowest_ind = i

        highest_tip = "t{}".format(highest_ind+tip_label_start)
        lowest_tip = "t{}".format(lowest_ind+tip_label_start)
        path = []
        path = any_path(G, highest_tip, lowest_tip)
        print("final path: ", path)
        spine_mask = skeleton/2
        prev = None
        for point in path:
            if('t' in point):
                label = int(point[1:])
                tip = tip_coordinates[label - tip_label_start]
                spine_mask[tip[0], tip[1]] = 255
            if('b' in point):
                label = int(point[1:])
                branch = branch_coordinates[label - branch_label_start]
                spine_mask[branch[0], branch[1]] = 255
            if(prev != None):
                edge = G.get_edge_data(prev, point)
                segment_label = int(edge["label"])
                for pixel in component_coordinates[segment_label-1]:
                    spine_mask[pixel[0], pixel[1]] = 255
            prev = point
        plt.imshow(spine_mask)
        plt.title("spine mask")
        plt.show()

    
        # print("nodes\n", nodes)

            # cv.imshow("skeleton", skeleton)

        # base_options = python.BaseOptions(model_asset_path="./model/deeplabv3.tflite")
        # options = vision.ImageSegmenterOptions(base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE, output_category_mask=True)

        # with vision.ImageSegmenter.create_from_options(options) as segmenter:
        #     for image_file_name in images:
        #         image = cv.imread(image_file_name)
        #         image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        #         segmentation_result = segmenter.segment(image)
        #         category_mask = segmentation_result.category_mask

        #         from collections import Counter
        #         print(Counter(category_mask.numpy_view().flatten()))
                
        #         image_data = image.numpy_view()
        #         fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        #         fg_image[:] = MASK_COLOR
        #         bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        #         bg_image[:] = BG_COLOR
                
        #         condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 10
        #         output_image = np.where(condition, image.numpy_view(), bg_image)
                
        #         print(f"Segmentation mask of {image_file_name}:")
        #         resize_and_show(image_file_name, output_image)
