import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

from tabsyn.nflow.model import Model_NFLOW_nextgen, Encoder_NFLOW_model_nextgen, Decoder_NFLOW_model_nextgen, Reconstructor
from utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def get_input_train(args):
    dataname = args.dataname
    latent = 'nflow'


    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/{latent}'
    train_embedding_save_path = f'{curr_dir}/ckpt/{dataname}/train_ae.npy'
    train_ae = torch.tensor(np.load(train_embedding_save_path)).float()
    
    train_ae = train_ae[:, 1:, :]
    B, num_tokens, token_dim = train_ae.size()
    in_dim = num_tokens * token_dim
    train_ae = train_ae.view(B, in_dim)

    test_embedding_save_path = f'{curr_dir}/ckpt/{dataname}/test_ae.npy'
    test_ae = torch.tensor(np.load(test_embedding_save_path)).float()
    
    test_ae = test_ae[:, 1:, :]
    B, num_tokens, token_dim = test_ae.size()
    in_dim = num_tokens * token_dim
    test_ae = test_ae.view(B, in_dim)
    return train_ae, test_ae, curr_dir, dataset_dir, ckpt_dir

def main(args):
    dataname = args.dataname
    latent = args.latent
    data_dir = f'data/{dataname}'

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    device =  args.device

    train_ae, test_ae,_, _, ckpt_path = get_input_train(args)

    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    _, _, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'])
    
    model_save_path = f'{ckpt_dir}/model.pt'
    #ae_encoder_load_path = f'{ckpt_dir}/ae_encoder.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    ae_decoder_load_path = f'{ckpt_dir}/ae_decoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    batch_size = 4096
    print('train_ae shape = ', train_ae.shape)
    train_loader = DataLoader(
        train_ae,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    pre_trained_decoder_model = Reconstructor(d_numerical, categories, D_TOKEN)
    pre_trained_decoder_model.eval()
    pre_trained_decoder_model.load_state_dict(torch.load(ae_decoder_load_path))
    
    model = Model_NFLOW_nextgen(d_numerical, categories, D_TOKEN, bias = True)
    
    test_ae = test_ae.to(device)
    # if torch.cuda.device_count() > 1:
        # device_ids = [0, 1, 2, 3]  # List of GPU IDs that you want to use
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model, device_ids = device_ids)
        # X_test_num = X_test_num.float().to('cuda')
        # X_test_cat = X_test_cat.to('cuda')
        # model.to('cuda:0')
    # else:
        # model.to(device)
        
    model.to(device)

    pre_encoder = Encoder_NFLOW_model_nextgen(d_numerical, categories, D_TOKEN).to(device)
    pre_decoder = Decoder_NFLOW_model_nextgen(d_numerical, categories, D_TOKEN).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    num_epochs = 100
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = 0.5 # lets give equal weights to reconstruction and log prob
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_log_loss = 0.0

        curr_count = 0

        for batch_data in pbar:
            model.train()
            optimizer.zero_grad()

            #batch_data = batch_data.numpy()
            #print('batch_data shape = ', batch_data.shape)
            #print('batch_data 0 = ', batch_data[0:2])
            
            batch_data = batch_data.to(device)
            log_prob = model.log_prob(batch_data)

            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            batch_length = batch_data.shape[0]
            curr_count += batch_length

            curr_log_loss += loss * batch_length

        nflow_loss = curr_log_loss / curr_count
        
        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            test_log_prob = model.log_prob(test_ae)

            test_loss = -test_log_prob.mean()
            #scheduler.step(test_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = test_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd


        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print('epoch: {}, Train NFLOW log prob:{:.6f}, Current Test log prob:{:.6f}'.format(epoch,  nflow_loss, test_log_prob.mean()))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    with torch.no_grad():
        
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model, pre_trained_decoder_model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        train_ae = train_ae.to(device)

        print('Successfully load and save the model!')

        train_z = pre_encoder(train_ae).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'