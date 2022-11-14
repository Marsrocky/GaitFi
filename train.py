import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from functions import labels2cat, Dataset_CRNN, train
from model import CRNN
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tripletloss import TripletLoss
import random
import argparse
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_path', type=str, default='./datasets/Train/image')
    parser.add_argument('--train_mat_path', type=str, default='./datasets/Train/mat')
    parser.add_argument('--test_image_path', type=str, default='./datasets/Test/image')
    parser.add_argument('--test_mat_path', type=str, default='./datasets/Test/mat')
    parser.add_argument('--save_model_path', type=str, default='./save_models/')
    parser.add_argument('--CNN_fc_hidden1', type=int, default=64)
    parser.add_argument('--CNN_fc_hidden2', type=int, default=64)
    parser.add_argument('--CNN_embed_dim', type=int, default=64)
    parser.add_argument('--img_x', type=int, default=64)
    parser.add_argument('--img_y', type=int, default=64)
    parser.add_argument('--dropout_p', type=float, default=0.4)
    parser.add_argument('--RNN_hidden_layers', type=int, default=1)
    parser.add_argument('--RNN_hidden_nodes', type=int, default=64)
    parser.add_argument('--RNN_FC_dim', type=int, default=64)
    parser.add_argument('--k', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--n_frames', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_type', type=str, default='both', choices=['image', 'mat', 'both'])
    parser.add_argument('--seed', type=int, default=233)
    args = parser.parse_args()

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
                                   shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

    # import mat shape
    mat_x, mat_y = 114, 500

    # Create model
    model = CRNN(args.img_x, args.img_y, mat_x, mat_y, args.CNN_fc_hidden1,
                 args.CNN_fc_hidden2, args.CNN_embed_dim, args.RNN_hidden_layers,
                 args.RNN_hidden_nodes, args.RNN_FC_dim, args.dropout_p, args.k, args.input_type).to(device)

    metric_loss = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # start training
    best_valid_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(args.epochs):
        # train, test model
        # model.train()
        train_loss = train(model, device, train_loader, optimizer, metric_loss, args.alpha)
        print('Epoch:{} train_loss:{:.6f}'.format(epoch + 1, train_loss))

    # save Pytorch models of best record
    torch.save(model, os.path.join(args.save_model_path,
                            'crnn_best_{}.pt'.format(args.input_type)))  # save best model




