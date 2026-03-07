import numpy as np
import torch
import pickle
from tqdm import tqdm
from itertools import repeat
from sklearn.metrics import mean_squared_error
import multiprocessing as mp
from multiprocessing import Pool

from src.laplace_models import NeuralNetworkRegression
from src.dual_models import DualGPRegression
from src.variational_models import VariationalNeuralRegression


def sample_data(ps=None, sigma_low=0.1, sigma_high=1., n_samples=50):
    xs, ys = list(), list()
    for i in range(n_samples):
        opt = np.random.choice(range(3), p=ps)
        if opt == 0:  # cluster left
            x = (np.random.rand() * 2) - 3.5  # from -3 to -1
            y = np.sin(x) + np.random.randn() * sigma_low
        elif opt == 1:  # cluster mid
            x = (np.random.rand() * 2) - 1  # from -1 to 1
            y = np.sin(x) + np.random.randn() * sigma_high
        elif opt == 2:  # cluster right
            x = (np.random.rand() * 2) + 1.5  # from 1 to 3
            y = np.sin(x) + np.random.randn() * sigma_low
        xs.append(x), ys.append(y)
    return np.array(xs), np.array(ys)


def get_datasets(k, ps, sl, sh, n, n_test=2000):
    ds = list()
    for _ in range(k):
        x, y = sample_data(ps=ps, n_samples=n, sigma_low=sl, sigma_high=sh)
        x = np.stack([x, np.ones(n)]).T
        xregion = np.stack([np.linspace(-3.5, 3.5, 2*n), np.ones(2*n)]).T
        yregion = np.sin(xregion[:, 0])
        x_test, y_test = sample_data(ps=ps, n_samples=n_test, sigma_low=sl, sigma_high=sh)
        x_test = np.stack([x_test, np.ones(n_test)]).T
        ds.append((xregion, yregion, x, y, x_test, y_test))
    return ds


def compute_marglik(delta, Ds, hidden_size, n_layers, sn, n_epochs=10000, vi_epochs=5000, act='tanh'):
    dres = {'fits': list(), 'mlh': list(), 'vimlh': list(), 'convimlh': list(), 'train_elbo': list(),
            'test_loss_map': list(), 'test_loss_lap': list(), 'test_loss_vi': list(), 'train_exp_lh': list(),
            'train_loss_map': list(), 'train_loss_lap': list(), 'train_loss_vi': list(), 'train_kl': list()}
    for j, (x, y, x_train, y_train, x_test, y_test) in enumerate(Ds):
        primal_nn = NeuralNetworkRegression(x_train, y_train, delta, sigma_noise=sn, n_epochs=n_epochs,
                                            hidden_size=hidden_size, activation=act,
                                            n_layers=n_layers, diagonal=True, n_samples_pred=1000, step_size=0.5)
        m_0, S_0 = np.zeros(primal_nn.d), 1 / delta * np.eye(primal_nn.d)
        (Us, Ss), vs = primal_nn.UsSs('J'), primal_nn.vs('J')
        x_hat, y_hat, s_noise = Us, Us @ primal_nn.theta_star - vs / Ss, 1 / np.sqrt(Ss)
        Us_full, _ = primal_nn.UsSs('J', x, y)
        dual_gp = DualGPRegression(x_hat, y_hat, s_noise, m_0, S_0=S_0, comp_post=True)
        lap_pred_train = primal_nn.posterior_predictive_f(x_train, 'J', compute_cov=False)
        map_pred_train = primal_nn.predictive_map(x_train)
        lap_pred_test = primal_nn.posterior_predictive_f(x_test, 'J', compute_cov=False)
        map_pred_test = primal_nn.predictive_map(x_test)
        # Variational Inference
        vi_nn = VariationalNeuralRegression(
            x_train,
            y_train,
            delta,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_epochs=vi_epochs,
        )
        vi_pred_train = vi_nn.posterior_predictive_f(x_train, compute_std=False)
        vi_pred_test = vi_nn.posterior_predictive_f(x_test, compute_std=False)
        dres['train_exp_lh'].append(vi_nn.loss)
        dres['train_kl'].append(vi_nn.KL)
        dres['train_elbo'].append(vi_nn.ELBO)
        dres['mlh'].append(dual_gp.log_marginal_likelihood())
        dres['vimlh'].append(vi_nn.compute_log_mlh())
        dres['convimlh'].append(vi_nn.compute_log_mlh_converged())
        dres['test_loss_map'].append(mean_squared_error(y_test, map_pred_test))
        dres['test_loss_lap'].append(mean_squared_error(y_test, lap_pred_test))
        dres['test_loss_vi'].append(mean_squared_error(y_test, vi_pred_test))
        dres['train_loss_map'].append(mean_squared_error(y_train, map_pred_train))
        dres['train_loss_lap'].append(mean_squared_error(y_train, lap_pred_train))
        dres['train_loss_vi'].append(mean_squared_error(y_train, vi_pred_train))
        map_pred = primal_nn.predictive_map(x)
        lap_pred, lp_cov = primal_nn.posterior_predictive_f(x, 'J', compute_cov=True)
        gp_pred, gp_cov = dual_gp.posterior_predictive_f(Us_full)
        lap_nn_err = np.sqrt(np.clip(np.diag(lp_cov), a_min=0, a_max=None))
        lap_gp_err = np.sqrt(np.clip(np.diag(gp_cov), a_min=0, a_max=None))
        vi_pred, vi_err = vi_nn.posterior_predictive_f(x, compute_std=True)
        fit = dict()
        fit['map'] = map_pred
        fit['lap'] = lap_pred
        fit['nncov'] = lap_nn_err
        fit['gpcov'] = lap_gp_err
        fit['vi'] = vi_pred
        fit['vicov'] = vi_err
        dres['fits'].append(fit)
    return dres


def reg_ms_delta(
    n_samples=100,
    n_retries=10,
    hidden_size=20,
    n_layers=2,
    n_params=21,
    n_epochs=10000,
    vi_epochs=5000,
    fname='',
    n_processes=None,
                 sigma_high=1, sigma_low=0.1, pm=0.45, po=0.1):
    sigma_low, sigma_high = sigma_low, sigma_high
    ps = [pm, po, pm]
    sn = 1
    Ds = get_datasets(n_retries, ps, sigma_low, sigma_high, n_samples)
    deltas = list(np.logspace(-2, 2, n_params))
    res = {'datasets': [{'x': x, 'y': y, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
                        for x, y, x_train, y_train, x_test, y_test in Ds], 'params': deltas}
    parameters = [(delta, datasets, hidden_size, n_layers, sn, n_epochs, vi_epochs) for delta, datasets in
                  zip(deltas, repeat(Ds))]
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    if n_processes == 1:
        metric_results = [compute_marglik(*param) for param in tqdm(parameters, total=len(deltas))]
    else:
        with Pool(processes=n_processes) as p:
            metric_results = tqdm(p.starmap(compute_marglik, parameters), total=len(deltas))

    res['results'] = list(metric_results)

    with open('results/reg_ms_delta_{fname}.pkl'.format(fname=fname), 'wb') as f:
        pickle.dump(res, f)
    return res


def reg_ms_width(
    n_samples=100,
    n_retries=10,
    delta=0.63,
    n_layers=2,
    n_params=30,
    n_epochs=10000,
    vi_epochs=5000,
    width_log_max=3.0,
    fname='',
    n_processes=None,
                 sigma_high=1, sigma_low=0.1, pm=0.45, po=0.1):
    sigma_low, sigma_high = sigma_low, sigma_high
    ps = [pm, po, pm]
    sn = 1
    Ds = get_datasets(n_retries, ps, sigma_low, sigma_high, n_samples)
    h_sizes = list(np.unique(np.logspace(0, width_log_max, n_params).astype(int)))
    res = {'datasets': [{'x': x, 'y': y, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
                        for x, y, x_train, y_train, x_test, y_test in Ds], 'params': h_sizes}
    parameters = [(delta, datasets, hsize, n_layers, sn, n_epochs, vi_epochs) for hsize, datasets in
                  zip(h_sizes, repeat(Ds))]
    print(len(parameters))
    print(h_sizes)
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    if n_processes == 1:
        metric_results = [compute_marglik(*param) for param in tqdm(parameters, total=len(h_sizes))]
    else:
        with Pool(processes=n_processes) as p:
            metric_results = tqdm(p.starmap(compute_marglik, parameters), total=len(h_sizes))

    res['results'] = list(metric_results)

    with open('results/reg_ms_width_{fname}.pkl'.format(fname=fname), 'wb') as f:
        pickle.dump(res, f)
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model selection experiment with result saving and MP.')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'delta', 'width'])
    parser.add_argument('--n_retries', type=int, default=10)
    parser.add_argument('--n_params', type=int, default=21)
    parser.add_argument('--width_n_params', type=int, default=30)
    parser.add_argument('--width_log_max', type=float, default=3.0)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--vi_epochs', type=int, default=5000)
    parser.add_argument('--sigma_lo', type=float, default=0.1)
    parser.add_argument('--sigma_hi', type=float, default=1)
    parser.add_argument('--pout', type=float, default=0.1)
    parser.add_argument('--pin', type=float, default=0.45)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n_processes', type=int, default=0, help='0 means auto (cpu_count-1), 1 disables multiprocessing')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_processes = None if args.n_processes == 0 else args.n_processes

    if args.mode in ('all', 'delta'):
        reg_ms_delta(
            fname=args.name,
            sigma_low=args.sigma_lo,
            sigma_high=args.sigma_hi,
            po=args.pout,
            pm=args.pin,
            n_retries=args.n_retries,
            n_params=args.n_params,
            n_epochs=args.n_epochs,
            vi_epochs=args.vi_epochs,
            n_processes=n_processes,
        )

    if args.mode in ('all', 'width'):
        reg_ms_width(
            fname=args.name,
            sigma_low=args.sigma_lo,
            sigma_high=args.sigma_hi,
            po=args.pout,
            pm=args.pin,
            n_retries=args.n_retries,
            n_params=args.width_n_params,
            n_epochs=args.n_epochs,
            vi_epochs=args.vi_epochs,
            width_log_max=args.width_log_max,
            n_processes=n_processes,
        )
    # TODO: delta, sigma, width for UCI data sets here or in new file!
