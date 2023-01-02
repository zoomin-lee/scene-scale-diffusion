from prettytable import PrettyTable
import torch
import os
import pickle
import numpy as np
import torch.nn.functional as F
import open3d as o3d

def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table

def get_miou_table(args, label_to_names, miou):
    table = PrettyTable(['Label', 'mIoU'])
    for i in range(args.num_classes):
        table.add_row([label_to_names[i], 100 * miou[i]])
    return table

def get_metric_table(metric_dict, epochs):
    table = PrettyTable()
    table.add_column('Epoch', epochs)
    if len(metric_dict)>0:
        for metric_name, metric_values in metric_dict.items():
            table.add_column(metric_name, metric_values)
    return table

def create_folders(args):
    # Create log folder
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.log_path+'/Completion', exist_ok=True)
    os.makedirs(args.log_path+'/Input', exist_ok=True)
    os.makedirs(args.log_path+'/Output', exist_ok=True)
    os.makedirs(args.log_path+'/Invalid', exist_ok=True)
    print("Storing logs in:", args.log_path)

def inter_vis(args, recons):
    for r in range(len(recons)):
        for batch, samples_i in enumerate(recons[r]):
            color_index = []
            for i in range(1, args.num_classes):
                index = torch.nonzero(samples_i == i ,as_tuple=False)
                color_index.append(F.pad(index,(1,0),'constant',value = i))
            colors_indexs = torch.cat(color_index, dim = 0).cpu().numpy()
            np.savetxt('/home/jumin/multinomial_diffusion/Result/Condition/Completion/iteration/batch{}_{}.txt'.format(batch, r), colors_indexs)


def visualization(args, recons, input_data, output, invalid, iteration):

    for batch, (samples_i, input_i, output_i, invalid_i) in enumerate(zip(recons, input_data, output, invalid)):
        color_index = []
        output_index = []
        input_points = torch.nonzero(input_i == 1, as_tuple=False).cpu().numpy()
        if args.dataset =='carla':
            invalid_points = torch.nonzero(invalid_i == 0, as_tuple=False).cpu().numpy() 
        elif args.dataset =='kitti':
            invalid_points = torch.nonzero(invalid_i == 1, as_tuple=False).cpu().numpy() 

        for i in range(1, args.num_classes):
            index = torch.nonzero(samples_i == i ,as_tuple=False)
            out_color = torch.nonzero(output_i == i, as_tuple=False)
            color_index.append(F.pad(index,(1,0),'constant',value = i))
            output_index.append(F.pad(out_color,(1,0),'constant',value=i))
        colors_indexs = torch.cat(color_index, dim = 0).cpu().numpy()
        out_indexs = torch.cat(output_index, dim = 0).cpu().numpy()
        np.savetxt(args.log_path+'/Completion/result_{}.txt'.format((iteration * args.batch_size) + batch), colors_indexs)

        '''np.savetxt(args.log_path+'/Input/input_{}.txt'.format((iteration * args.batch_size) + batch), input_points)
        np.savetxt(args.log_path+'/Invalid/invalid_{}.txt'.format((iteration * args.batch_size) + batch), invalid_points)
        np.savetxt(args.log_path+'/Output/gt_{}.txt'.format((iteration * args.batch_size) + batch), out_indexs)'''
        

def completion_vis(args, input_p, recons):
    for batch, (recon_i, input_i) in enumerate(zip(recons, input_p)):
        recon_points = torch.nonzero(recon_i == 1, as_tuple=False).cpu().numpy()
        input_points = torch.nonzero(input_i == 1, as_tuple=False).cpu().numpy()
        np.savetxt(args.log_path+'/Completion/completion_{}.txt'.format(batch), recon_points)
        np.savetxt(args.log_path+'/Input/input_{}.txt'.format(batch), input_points)


def iou_one_frame(pred, target, n_classes=23):
    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        intersection[cls] = np.sum((pred == cls) & (target == cls))
        union[cls] = np.sum((pred == cls) | (target == cls))
    return intersection, union


def get_result(args, for_mask, output, preds, SSC=True):
    for_mask = for_mask.contiguous().view(-1)
    output = output.contiguous().view(-1)
    preds = preds.contiguous().view(-1)
    
    if SSC :
        if args.dataset == 'kitti':
            mask = for_mask == 0
        elif args.dataset== 'carla':
            mask = for_mask > 0
    else : 
        mask = for_mask == 1

    output_masked = output[mask]
    iou_output_masked = output_masked.cpu().numpy()
    iou_output_masked[iou_output_masked != 0] = 1

    preds_masked = preds[mask]
    iou_preds_masked = preds_masked.cpu().numpy()
    iou_preds_masked[iou_preds_masked != 0] = 1

    # I, U for a frame
    correct = np.sum(output_masked.cpu().numpy() == preds_masked.cpu().numpy())
    total = preds_masked.shape[0]

    pred_TP = np.sum((iou_preds_masked == 1) & (iou_output_masked == 1))
    pred_FP = np.sum((iou_preds_masked == 1) & (iou_output_masked == 0))
    pred_TN = np.sum((iou_preds_masked == 0) & (iou_output_masked == 0))
    pred_FN = np.sum((iou_preds_masked == 0) & (iou_output_masked == 1))

    intersection, union = iou_one_frame(preds_masked, output_masked, n_classes=args.num_classes)
    return correct, total, pred_TP, pred_FP, pred_TN, pred_FN, intersection, union

def save_args(args):
    # Save args
    with open(os.path.join(args.log_path, 'args.pickle'), "wb") as f:
        pickle.dump(args, f)

    # Save args table
    args_table = get_args_table(vars(args))
    with open(os.path.join(args.log_path,'args_table.txt'), "w") as f:
        f.write(str(args_table))

def print_completion(num_correct, num_total, TP, FP, FN):
    print("\n=========================================\n")
    accuracy = num_correct/num_total
    print("\nAccuracy : ", accuracy)

    precision = 100 * TP / (TP + FP)
    recall = 100 * TP / (TP + FN)
    iou = 100 * TP / (TP + FP + FN)

    print("\nCompleteness")
    print("precision:", precision)
    print("recall:", recall)
    print("iou:", iou)

    print("\n=========================================\n")
    return iou

def print_result(args, label_to_names, num_correct, num_total, all_intersections, all_unions, TP, FP, FN, SSC=True):
    if SSC :
        print("\n========== Semantic Scene Completion =============\n")
    else :
        print("\n============ Semantic Segmentation ===============\n")
    accuracy = num_correct/num_total
    print("\nAccuracy : ", accuracy)

    precision = 100 * TP / (TP + FP)
    recall = 100 * TP / (TP + FN)
    iou = 100 * TP / (TP + FP + FN)

    print("\nCompleteness")
    print("precision:", precision)
    print("recall:", recall)
    print("iou:", iou)

    print("\nSemantic IoU Per Class")
    miou = all_intersections / all_unions
    for i in range(args.num_classes):
        print(label_to_names[i], ':', 100 * miou[i])
    print("\n====================================================\n")
    return iou, miou