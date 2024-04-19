"""
Methods for formatted output.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""
import os

from copy import deepcopy

def format_str_submission(roll, pitch, yaw, x, y, z, score):
    """
    Get a prediction string in ApolloScape style.
    """      
    tempt_str = "{pitch:.3f} {yaw:.3f} {roll:.3f} {x:.3f} {y:.3f} {z:.3f} {score:.3f}".format(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            x=x,
            y=y,
            z=z,
            score=score)
    return tempt_str

def get_instance_str(dic):
    """
    Produce KITTI style prediction string for one instance.
    """     
    string = ""
    string += dic['class'] + " "
    string += "{:.1f} ".format(dic['truncation'])
    string += "{:.1f} ".format(dic['occlusion'])
    string += "{:.6f} ".format(dic['alpha'])
    string += "{:.6f} {:.6f} {:.6f} {:.6f} ".format(dic['bbox'][0], dic['bbox'][1], dic['bbox'][2], dic['bbox'][3])
    string += "{:.6f} {:.6f} {:.6f} ".format(dic['dimensions'][1], dic['dimensions'][2], dic['dimensions'][0])
    string += "{:.6f} {:.6f} {:.6f} ".format(dic['locations'][0], dic['locations'][1], dic['locations'][2])
    string += "{:.6f} ".format(dic['rot_y'])
    if 'score' in dic:
        string += "{:.8f} ".format(dic['score'])
    else:
        string += "{:.8f} ".format(1.0)
    return string

def get_pred_str(record):
    """
    Produce KITTI style prediction string for a record dictionary.
    """      
    # replace the rotation predictions of input bounding boxes
    updated_txt = deepcopy(record['raw_txt_format'])
    for instance_id in range(len(record['euler_angles'])):
        updated_txt[instance_id]['rot_y'] = record['euler_angles'][instance_id, 1]
        updated_txt[instance_id]['alpha'] = record['alphas'][instance_id]
    pred_str = ""
    angles = record['euler_angles']
    for instance_id in range(len(angles)):
        # format a string for submission
        tempt_str = get_instance_str(updated_txt[instance_id])
        if instance_id != len(angles) - 1:
            tempt_str += '\n'
        pred_str += tempt_str
    return pred_str

def save_txt_file(img_path, prediction, params):
    """
    Save a txt file for predictions of an image.
    """    
    if not params['flag']:
        return
    file_name = img_path.split('/')[-1][:-3] + 'txt'
    save_path = os.path.join(params['save_dir'], file_name) 
    with open(save_path, 'w') as f:
        f.write(prediction['pred_str'])
    print('Wrote prediction file at {:s}'.format(save_path))
    return