import os
import random
import dgl
import time
import gc
import torch.cuda
from modules.DNN import DNN
from modules.VGAE import Model
from time import gmtime, strftime
from modules import diffusion as gd
from modules.generator import Graph_Editer

from parameters import args
from utils.evaulate import compute_vgae_loss, adjust_loss, compute_loss_para
from utils.input_data import UserItemDataset
from utils.preprocess import mask_test_edges_dgl
from torch.utils.data import DataLoader
from utils.util_loss import *
from modules.rec_model import LGCN_Encoder
from modules.environment_inference import *

torch.autograd.set_detect_anomaly(False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Keep launch blocking off for speed/memory during full reproduction runs.


def setup_device():
    return torch.device("cuda:{}".format(0))


def load_datasets(data_name):
    dataset_paths = {
        "yelp2018": {
            "train": "./dataset/yelp2018/yelp_train_data.bin",
            "test": "./dataset/yelp2018/yelp_test_data.bin",
        },
        "douban": {
            "train": "./dataset/douban/douban_train_data.bin",
            "test": "./dataset/douban/douban_test_data.bin",
        }  # Please add your own dataset path in there, and you need first add you dataset into ./dataset.
    }

    if data_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {data_name}")

    selected_paths = dataset_paths[data_name]
    datasets = {key: dgl.load_graphs(path)[0][0] for key, path in selected_paths.items()}
    return datasets


def prepare_data(graph, device):
    num_nodes = graph.number_of_nodes()
    feats = graph.ndata['feat'].to(device)
    adj_orig = None
    edge_index = torch.stack(graph.edges())
    return feats, adj_orig, edge_index, num_nodes


def initialize_models(num_nodes, device, in_dim, mlp_in_dims, mlp_out_dims):
    vgae_model = Model(in_dim, args.hidden1, args.hidden2, device, num_nodes).to(device)
    diffusion_model = gd.GaussianDiffusion(gd.ModelMeanType.START_X,
                                           args.noise_schedule, args.noise_scale, args.noise_min,
                                           args.noise_max, args.steps, device).to(device)
    mlp_model = DNN(mlp_in_dims, mlp_out_dims, args.emb_size, env_size=args.hidden2, time_type="cat",
                    norm=args.norm, act_func=args.mlp_act_func).to(device)
    generator = Graph_Editer(args.env_k, num_nodes, device).to(device)
    env_infer_model = EVAE(args.hidden2, args.hidden2).to(device)

    # mlp_num = sum([param.nelement() for param in mlp_model.parameters()])
    # diff_num = sum([param.nelement() for param in diffusion_model.parameters()])
    # vgae_num = sum([p.nelement() for p in vgae_model.parameters()])
    # env_infer_num = sum([p.nelement() for p in env_infer_model.parameters()])

    # params = mlp_num + diff_num + vgae_num + env_infer_num
    # print('Total Parameters:', params)
    return vgae_model, diffusion_model, mlp_model, generator, env_infer_model


def setup_optimizers(models):
    lr = args.learning_rate
    lr2 = args.lr2
    wd2 = args.wd2
    optimizers = [
        torch.optim.Adam(models[0].parameters(), lr=lr),
        torch.optim.Adagrad(models[2].parameters(), lr=lr2, weight_decay=wd2),
        # SGD for the (K,n,n) editor: Adagrad keeps an extra ~4GB state on Yelp and OOMs on 24GB.
        # Paper Alg.1 uses AdamW for other modules; released code used Adagrad for the editor.
        torch.optim.SGD(models[3].parameters(), lr=lr),
        torch.optim.Adagrad(models[4].parameters(), lr=lr)
    ]
    return optimizers

def train_model(model, optimizers, device, datasets, user_item_train_inter, num_user, num_item):
    global best_epoch
    graph = datasets['train']
    train_edge_idx = mask_test_edges_dgl(graph)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False).to(device)
    adj = train_graph.adjacency_matrix().to_dense().to(device)
    weight_tensor, norm = compute_loss_para(adj, device)
    feats, _, edge_index, _ = prepare_data(graph, device)
    bestPerformance = []
    measure_result = {}
    for epoch in range(args.epochs):
        total_loss, rec_loss = run_epoch(model, optimizers, feats, edge_index, adj, norm, weight_tensor,
                                         device, epoch, user_item_train_inter, num_user, num_item, train_graph)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss}, Rec_loss: {rec_loss}")

        measure, best_epoch = test_epoch(datasets, epoch, model, device, bestPerformance, num_user, num_item)
        measure_result[epoch] = measure
        torch.cuda.empty_cache()

    print('The best result of %s:\n%s' % ('causal', ''.join(measure_result[best_epoch - 1])))


def run_epoch(models, optimizers, feats, edge_index, adj, norm, weight_tensor, device, epoch, user_item_train_inter,
              num_user, num_item, train_graph):
    vgae_model, diffusion_model, mlp_model, generator, env_infer_model = models
    mlp_model.train()

    pretrain_loss, total_rec_loss = 0.0, 0.0

    generator.reset_parameters()
    beta = compute_beta(epoch, args.epochs)

    for m in range(1):
        # Two-pass EERM to keep only one environment graph in memory (fits 24GB):
        # Pass 1: collect detached losses & log_p for mean/var coefficients
        # Pass 2: re-forward each env and backward scaled scalar loss
        detached_losses = []
        detached_logps = []
        for k in range(args.env_num):
            with torch.no_grad():
                loss_k, log_p_k = _env_forward(
                    vgae_model, diffusion_model, mlp_model, generator, env_infer_model,
                    feats, edge_index, adj, norm, weight_tensor, k)
            detached_losses.append(loss_k.item())
            detached_logps.append(log_p_k.detach())
            torch.cuda.empty_cache()

        loss_t = torch.tensor(detached_losses, device=device, dtype=torch.float64)
        mean_v = loss_t.mean()
        # Coefficients matching d/dL_i of (Var + beta * Mean) under sample var
        # Var = sum((L_i - mean)^2) / (n-1), Mean = sum L_i / n
        n_env = max(args.env_num, 1)
        coefs = []
        for i in range(n_env):
            if n_env > 1:
                var_coef = 2.0 * (detached_losses[i] - mean_v.item()) / (n_env - 1)
            else:
                var_coef = 0.0
            coefs.append(var_coef + beta / n_env)

        optimizer1, optimizer2, optimizer3, optimizer4 = optimizers
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer4.zero_grad()

        for k in range(args.env_num):
            loss_k, _ = _env_forward(
                vgae_model, diffusion_model, mlp_model, generator, env_infer_model,
                feats, edge_index, adj, norm, weight_tensor, k)
            (coefs[k] * loss_k).backward()
            del loss_k
            torch.cuda.empty_cache()

        if m == 0:
            optimizer1.step()
            optimizer2.step()
            optimizer4.step()

        # Generator inner step: maximize env diversity reward (use loss variance)
        var_reward = loss_t.var(unbiased=False).detach() if n_env > 1 else loss_t.new_zeros(())
        optimizer3.zero_grad()
        Log_p = 0
        for k in range(args.env_num):
            _, log_p_k = generator(feats.shape[0], 5, k, edge_index, noise_level=args.edit_noise)
            Log_p = Log_p + log_p_k
        inner_loss = -(var_reward * Log_p).mean()
        inner_loss.backward()
        optimizer3.step()

        outer_loss_val = (loss_t.var(unbiased=False) + mean_v * beta).item() if n_env > 1 else (mean_v * beta).item()
        print(torch.tensor(outer_loss_val, device=device))
        pretrain_loss += outer_loss_val
        torch.cuda.empty_cache()

    all_embeddings = generate_embeddings(models[0], models[1], models[2], models[4], feats, edge_index, 1, device)
    user_embeddings = all_embeddings[:num_user]
    item_embeddings = all_embeddings[num_user:]

    ui_adj = generate_interaction_matrix_from_dgl(train_graph, num_user, num_item)
    norm_adj = normalize_graph_mat(ui_adj)

    model = LGCN_Encoder(num_user, 3, norm_adj, user_embeddings, item_embeddings)
    rec_model = model.to(device)
    optimizer = torch.optim.Adam(rec_model.parameters(), lr=0.001)

    dataset = UserItemDataset(user_item_train_inter)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)  # Reduce batch size
    for batch in dataloader:
        user_id, pos_item_id, neg_item_id = [x.to(device) for x in batch]
        all_embeddings = rec_model().to(device)

        user_embedding = all_embeddings[user_id]
        pos_item_embedding = all_embeddings[pos_item_id]
        neg_item_embedding = all_embeddings[neg_item_id]

        rec_loss = bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding) + \
                   l2_reg_loss(1e-3, user_embedding, pos_item_embedding, neg_item_embedding)
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()

        total_rec_loss += rec_loss.item()
        torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 0.7)

        torch.cuda.empty_cache()

    return pretrain_loss, total_rec_loss


def _env_forward(vgae_model, diffusion_model, mlp_model, generator, env_infer_model,
                 feats, edge_index, adj, norm, weight_tensor, k):
    dge_index, log_p = generator(feats.shape[0], 5, k, edge_index, noise_level=args.edit_noise)
    gc.collect()
    torch.cuda.empty_cache()
    batch_latent = vgae_model.encoder(feats, dge_index)
    del dge_index

    recon, mu, log_std = env_infer_model(batch_latent)
    kl_div = env_infer_model.kl_divergence(mu, log_std)
    infer_loss = evae_loss(recon, log_std, kl_div)
    env_embeddings = env_infer_model.decode(batch_latent)

    terms = diffusion_model.training_losses(mlp_model, batch_latent, env_embeddings, args.reweight)
    elbo = terms["loss"].mean()
    pred_xstart = terms["pred_xstart"]
    del terms
    logits = vgae_model.decoder(pred_xstart)
    del pred_xstart
    torch.cuda.empty_cache()
    vgae_loss = compute_vgae_loss(logits, adj, norm, vgae_model, weight_tensor)
    del logits
    torch.cuda.empty_cache()
    loss = adjust_loss(elbo, vgae_loss, infer_loss, args.reweight)
    return loss, log_p


def test_epoch(datasets, epoch, models, device, bestPerformance, num_user, item_user):
    test_graph = datasets['test']
    feats, _, edge_index, _ = prepare_data(test_graph, device)
    all_embeddings = generate_embeddings(models[0], models[1], models[2], models[4], feats, edge_index, 1, device)
    user_embeddings = all_embeddings[:num_user]
    item_embeddings = all_embeddings[num_user:]
    ui_adj = generate_interaction_matrix_from_dgl(test_graph, num_user, item_user)
    norm_adj = normalize_graph_mat(ui_adj)
    del ui_adj
    model = LGCN_Encoder(num_user, 3, norm_adj, user_embeddings, item_embeddings)
    rec_model = model.to(device)

    with torch.no_grad():
        all_embeddings = rec_model().to(device)
        user_embeddings = all_embeddings[:num_user]
        item_embeddings = all_embeddings[num_user:]

    scores = torch.matmul(user_embeddings, item_embeddings.t())

    origin_inter, user_set = get_origin_user_interaction_list(test_graph)
    rec_dict = get_rec_list(user_set, scores, num_user)
    # print("recommendation list")

    measure = ranking_evaluation(origin_inter, rec_dict, [10, 20])
    measure_index = measure.index('Top 20\n')
    measure_input = measure[measure_index:]
    best_epoch = fast_evaluation(epoch, measure_input, bestPerformance)
    print(best_epoch)

    return measure, best_epoch


def handle_gradient_step(optimizers, outer_loss, Log_p, m, Var):
    optimizer1, optimizer2, optimizer3, optimizer4 = optimizers
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    print(outer_loss)
    if m == 0:
        outer_loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer4.step()
    reward = Var.detach()
    inner_loss = - reward * Log_p
    inner_loss = inner_loss.mean()
    inner_loss.backward()
    optimizer3.step()


def generate_embeddings(vgae_model, diffusion_model, mlp_model, env_infer, features, edge_index, num_samples, device):
    features = features.to(device)
    edge_index = edge_index.to(device)
    vgae_model = vgae_model.to(device)
    mlp_model = mlp_model.to(device)
    env_infer = env_infer.to(device)

    embeddings_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            z = vgae_model.encoder(features, edge_index)
            env_embeddings = env_infer.decode(z)
            diffused_z = diffusion_model.p_sample(mlp_model, z, env_embeddings, args.sampling_steps,
                                                  args.sampling_noise)
            embeddings_list.append(diffused_z)

    final_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)

    return final_embeddings


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


if __name__ == '__main__':
    print("Start model training!")
    start_time = time.time()
    seed_it(1024)
    device = setup_device()

    # loading data
    datasets = load_datasets(args.dataset)
    train_graph = datasets['train']
    feats, adj_orig, edge_index, num_nodes = prepare_data(train_graph, device)
    indim = feats.shape[-1]
    latent_size = args.emd_size
    mlp_out_dims = eval(args.mlp_dims) + [latent_size]
    mlp_in_dims = mlp_out_dims[::-1]
    print(f"Reproduce settings: dataset={args.dataset}, epochs={args.epochs}, steps={args.steps}, "
          f"emd_size={args.emd_size}, lr={args.learning_rate}, env_k={args.env_k}, "
          f"env_num={args.env_num}, edit_noise={args.edit_noise}")
    models = initialize_models(datasets['train'].number_of_nodes(), device, indim, mlp_in_dims, mlp_out_dims)
    optimizers = setup_optimizers(models)

    user_item_train_inter, num_users, num_items = get_user_item_matrix(train_graph)
    train_model(models, optimizers, device, datasets, user_item_train_inter, num_users, num_items)
    end_time = time.time()
    run_time = strftime("%H:%M:%S", gmtime(end_time - start_time))
    print("total time cost：{0}".format(run_time))
