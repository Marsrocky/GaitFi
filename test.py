import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from functions import labels2cat, Dataset_CRNN, validation, acc_calculate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import random
import argparse
import warnings
import json

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_path', type=str, default='./datasets/Train/image')
    parser.add_argument('--train_mat_path', type=str, default='./datasets/Train/mat')
    parser.add_argument('--test_image_path', type=str, default='./datasets/Test/image')
    parser.add_argument('--test_mat_path', type=str, default='./datasets/Test/mat')
    parser.add_argument('--img_x', type=int, default=64)
    parser.add_argument('--img_y', type=int, default=64)
    parser.add_argument('--k', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_frames', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_type', type=str, default='both', choices=['image', 'mat', 'both'])
    parser.add_argument('--seed', type=int, default=233)
    args = parser.parse_args()

    args.load_model_path = f'./best_models/crnn_best_{args.input_type}.pt'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU

    # convert labels -> category
    action_names = os.listdir(args.train_image_path)

    le = LabelEncoder()
    le.fit(action_names)

    # show how many classes
    print('labels:{}'.format(list(le.classes_)))

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    train_actions = []
    train_all_names = []
    test_actions = []
    test_all_names = []
    for action in action_names:
        for f_name in os.listdir(f'{args.train_image_path}/{action}'):
            train_actions.append(action)
            train_all_names.append(f'{action}/{f_name}')

        for f_name in os.listdir(f'{args.test_image_path}/{action}'):
            test_actions.append(action)
            test_all_names.append(f'{action}/{f_name}')

    train_list = train_all_names
    train_label = labels2cat(le, train_actions)
    test_list = test_all_names  # all video file names
    test_label = labels2cat(le, test_actions)  # all video labels

    transform = transforms.Compose([transforms.Resize([args.img_x, args.img_y]), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])  # 串联多个图片变换

    train_set = Dataset_CRNN(args.train_image_path, args.train_mat_path,
                             train_list, train_label, args.n_frames, transform=transform, input_type=args.input_type)
    test_set = Dataset_CRNN(args.test_image_path, args.test_mat_path,
                            test_list, test_label, args.n_frames, transform=transform, input_type=args.input_type)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

    # Create model
    model = torch.load(args.load_model_path)

    gallery_feat, gallery_label, prob_feat, prob_label = validation(model, device, train_loader, test_loader)

    gallery_feat = torch.cat(gallery_feat)
    gallery_label = torch.cat(gallery_label)
    prob_feat = torch.cat(prob_feat)
    prob_label = torch.cat(prob_label)

    test_correct, test_total = acc_calculate(gallery_feat, gallery_label, prob_feat, prob_label)

    with open(f'./outputs/saved_outputs_{args.input_type}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps({
            'gallery_feat': gallery_feat.detach().cpu().numpy().tolist(),
            'gallery_label': gallery_label.detach().cpu().numpy().tolist(),
            'prob_feat': prob_feat.detach().cpu().numpy().tolist(),
            'prob_label': prob_label.detach().cpu().numpy().tolist()
        }))

    test_acc = test_correct / test_total * 100
    print('test_acc:{:.3f}%'.format(test_acc))




