import sys
import time
import os.path
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from scipy import linalg as la
from scipy import sparse as spa
from scipy.sparse import linalg as sla
from matplotlib import pyplot as plt
from collections import defaultdict


sp.set_printoptions(precision=8, suppress=True)
np.set_printoptions(precision=8)

global filename, overwrite, log, pagerank, node_list, elements, sorted_elements


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    with open(log, 'a') as file:
        print(*args, **kwargs, file=file)


def file_to_graph(delimiter):
    print_log('converting edge list to a graph...')
    df = pd.read_csv('Datasets/others/{0}'.format(filename), delimiter=delimiter, header=0)
    src = df.iloc[:, 0]  # Not a copy, just a reference.
    # print_log(src)
    trg = df.iloc[:, 1]
    # print_log(trg)
    graph = nx.DiGraph()
    # graph.add_nodes_from(range(count))
    graph.add_edges_from(list(zip(src, trg)))
    graph.add_edges_from([n, n] for n in graph)
    print_log('finished importing the graph. It has {0} nodes and {1} edges'.format(len(graph),
                                                                                    graph.number_of_edges()))
    return graph


def graph_to_adjacency_dicts(graph):

    """Converts the graph to a dictionary representation of every node and it's adjacent nodes"""
    adj = nx.to_dict_of_lists(graph)

    return adj


def graph_to_sparse_matrix(graph):

    """Converts the graph to a sparse matrix in coo format, reconvert if necessary"""
    sparse = nx.to_scipy_sparse_matrix(graph)
    sparse = sparse.tocsr()

    return sparse


def sort_elements():

    global pagerank, sorted_elements, elements
    elem_pagerank = defaultdict()

    for element in elements:
        print('element: ', element)
        print(node_list.index(element))
        elem_pagerank[element] = pagerank[node_list.index(element)]
    sorted_elements = sorted(elem_pagerank.items(), key=lambda kv: kv[1], reverse=True)


def get_sorted_matrix(matrix_in):

    global sorted_elements, elements
    matrix = np.array(matrix_in)
    print_log('sorting the matrix elements according to their pagerank values...')
    sorted_matrix = np.empty(matrix.shape, dtype=float)
    sorted_row_matrix = np.empty(matrix.shape, dtype=float)
    for i, (node, rank) in enumerate(sorted_elements):
        print_log(i, node, rank)
        node_id = elements.index(node)
        sorted_row_matrix[i, :] = matrix[node_id, :]
    for i, (node, rank) in enumerate(sorted_elements):
        node_id = elements.index(node)
        sorted_matrix[:, i] = sorted_row_matrix[:, node_id]
    print_log('finished sorting a matrix')
    # print_log(sorted_matrix)
    return sorted_matrix


def get_google_matrix(graph):
    print_log('creating a Google matrix...')
    google_matrix = np.array(nx.google_matrix(graph, nodelist=node_list))
    print_log(google_matrix)
    sparse_google_matrix = spa.csr_matrix(google_matrix)
    return sparse_google_matrix


def get_google_matrix_old(graph_in):

    print_log('creating google matrix')

    adj = graph_to_adjacency_dicts(graph_in)
    # get all indices for nonzero elements in a sparse matrix
    # [((i, j), a[i, j]) for i, j in zip(*a.nonzero())]

    # S = nx.Graph()
    graph = nx.Graph()
    num_nodes = len(adj)
    for node in adj:
        graph.add_node(node)
    alpha = 0.85
    for count, j in enumerate(adj):
        # print_log(count)
        out_edges = len(adj[j])
        k_out = out_edges if out_edges > 0 else 1 / num_nodes
        for i in adj:
            adj_ij = 1 if j in adj[i] else 0
            weight = alpha * adj_ij / k_out + (1 - alpha) / num_nodes
            graph.add_edge(i, j, weight=weight)
            # the next three lines are equivalent to the previous and can be used if S_ij is needed elsewhere
            # S[i][j]['weight'] = A_ij / k_out
            # S_ij = S[i][j]['weight']
            # G[i][j]['weight'] = alpha * S_ij + (1 - alpha) / N
    nx.write_adjlist(graph, "graph.adjlist")
    G = graph_to_sparse_matrix(graph).todense()
    # print_log('len(adj):', len(adj))
    return G


def get_eigen(g):

    print_log('calculating the eigenvalues...')
    l_ev, v_l, v_r = la.eig(g, left=True)
    # l_ev vector of eigenvalues
    # v_l left eigenvector
    # v_r right eigenvector

    # usage if only interested in the largest eigenvalue:
    v_r = v_r[:, 0].real
    v_l = v_l[:, 0].real
    l_c = l_ev[0]

    return l_c, v_l, v_r


def get_eigen_sparse(g):
    print_log('calculating the eigenvalues...')
    eig_r, v_r = sla.eigs(g, k=1, which='LR')
    eig_l, v_l = sla.eigs(g.transpose(), k=1, which='LR')
    # if not (eig_l == eig_r):
    #   print_log('the eigenvalues are different:')
    #   print_log(eig_l, eig_r)

    return eig_r.real, spa.csr_matrix(v_l.real), spa.csr_matrix(v_r.real)


def decompose_matrix(g, count):

    g_rr = g[0:count, 0:count]
    g_rs = g[0:count, count:]
    g_sr = g[count:, 0:count]
    g_ss = g[count:, count:]

    return g_rr.todense(), g_rs, g_sr, g_ss


def calc_g_pr(g_rs, g_sr, l, v_r, v_l):

    # the following two lines give equivalent results, might be different in performance
    # print_log('G_rs ', G_rs, '\n p_c:\n', P_c, '\n G_sr:\n', G_sr)
    # G_pr = (G_rs @ p_c @ G_sr)/(1-l)
    # print_log('G_rs ', G_rs, '\n v_r:\n', v_r, '\n G_sr:\n', G_sr, '\n v_l:\n', v_l)
    # print_log(g_rs.shape, v_r.shape)
    vr_x = (g_rs @ v_r).todense()
    v_l_row = v_l.T
    # print_log(v_l_row.shape, v_l_row.dtype, g_sr.shape, g_sr.dtype)
    vl_x = (v_l_row @ g_sr).todense()
    g_pr = (np.outer(vr_x, vl_x) / (1 - l)).real
    # print_log('\nvr_x:\n', vr_x, '\nvl_x:\n', vl_x, '\ng_pr:\n', g_pr)
    np.save('Matrices/g_pr_{0}.npy'.format(filename), g_pr)
    return g_pr


def calc_sum_expr(g_ss_x):

    sum_expr = np.identity(g_ss_x.shape[0], dtype=float)
    epsilon = np.ones(g_ss_x.shape) * 0.001
    prev = g_ss_x
    sum_expr += prev
    # print_log('sum_expr: \n', sum_expr, '\nprev:\n', prev, '\nepsilon:\n', epsilon)
    # print_log('\nsum_expr - prev:\n', np.abs(sum_expr - prev))
    # print_log((np.abs(sum_expr - prev) > epsilon).all())
    l = 1
    while not (prev < np.full(prev.shape, np.inf)).any() & (prev > epsilon).any():
        # print_log('\nsum_expr - prev:\n', np.abs(sum_expr - prev))
        # print_log('\nsum_expr:\n', sum_expr)
        prev = prev @ g_ss_x
        sum_expr += prev
        #  'sum_expr: ', sum_expr, '\nprev:\n', prev,
        # print_log('\nsum_expr:\n', sum_expr)
        l += 1
        print_log(l)
    np.save('Matrices/sum_expr_{0}.npy'.format(filename), sum_expr)
    return sum_expr


def calc_g_qr(g_ss, g_rs, g_sr):

    print_log('calculating g_qr...')
    print_log('calculating g_ss_x...')

    q_c = np.load('Matrices/q_c_{0}.npy'.format(filename))
    q_c_s = spa.csr_matrix(q_c)
    if overwrite or not os.path.exists('Matrices/sum_expr_{0}.npy'.format(filename)):
        g_ss_x = q_c_s @ g_ss @ q_c_s
        print_log(g_ss.shape, g_ss_x.shape)
        np.save('Matrices/g_ss_x_{0}.npy'.format(filename), g_ss_x)
        sum_expr = calc_sum_expr(g_ss_x)
    # g_ss_x = np.load('g_ss_x_{0}.npy'.format(filename))
    else:
        sum_expr = np.load('Matrices/sum_expr_{0}.npy'.format(filename))
    # print_log('\ng_ss:\n', g_ss, '\nq_c:\n', q_c, '\ng_ss_x:\n', g_ss_x)
    # print_log('g_ss_x: \n', g_ss_x, '\nmatrix_power(g_ss_x, 0):\n', np.linalg.matrix_power(g_ss_x, 0))
    print_log('calculating the sum...')
    print_log(sum_expr.shape, g_sr.shape)
    sum_expr_g_sr = sum_expr @ g_sr
    print_log('calculating the product...')
    prod = q_c_s @ sum_expr_g_sr
    g_qr = g_rs @ prod
    print_log('finished calculating g_qr')
    np.save('Matrices/g_qr_{0}.npy'.format(filename), g_qr)
    return g_qr


def calc_sum_expr_g_sj(sum_expr, col):

    sum_expr_g_sr_col = sum_expr @ col

    return sum_expr_g_sr_col


def calc_g_r(g, count):

    print_log('calculating g_r...')
    g_rr, g_rs, g_sr, g_ss = decompose_matrix(g, count)
    # print_log('\ng:\n', g, '\ng_rr:\n', g_rr, '\ng_rs:\n', g_rs, '\ng_sr:\n', g_sr, '\ng_ss:\n', g_ss)
    # l_c, v_l, v_r = get_eigen(g_ss.todense()
    try:
        l_c, v_l, v_r = get_eigen_sparse(g_ss)
    except:
        l_c, v_l, v_r = get_eigen(g_ss.todense())
    print_log('calculating q_c...')
    # index, columns = g_ss.shape
    v_l_c = spa.coo_matrix(v_l)
    v_r_c = spa.coo_matrix(v_r)
    # print_log(v_l.shape, v_r.shape)
    # q_c = v_l * v_r.T
    if overwrite or not os.path.exists('Matrices/q_c_{0}.npy'.format(filename)):
        q_c = np.empty(g_ss.shape, dtype='float64')
        for i, m, u in zip(v_l_c.row, v_l_c.col, v_l_c.data):
            for j, n, v in zip(v_r_c.row, v_r_c.col, v_r_c.data):
                value = u * v
                if i == j:
                    q_c[i][j] = 1 - value
                else:
                    q_c[i][j] = 0 - value
            # print_log('finished with row {0} of q_c'.format(i))
        np.save('Matrices/q_c_{0}.npy'.format(filename), q_c)
    else:
        q_c = np.load('Matrices/q_c_{0}.npy'.format(filename))

    print_log('calculating g_pr...')
    g_pr = calc_g_pr(g_rs, g_sr, l_c, v_l, v_r)
    print_log('calculating g_qr...')
    g_qr = calc_g_qr(g_ss, g_rs, g_sr)
    # print_log('\ng_rr: \n', g_rr, '\ng_pr: \n', g_pr, '\ng_qr: \n', g_qr)
    print_log('calculating g_r...')

    g_r = g_rr + g_pr + g_qr
    # for sorting the matrix, use this
    # sorted_g_rr = get_sorted_matrix(g_rr)
    # sorted_g_pr = get_sorted_matrix(g_pr)
    # sorted_g_qr = get_sorted_matrix(g_qr)
    # g_r = sorted_g_rr + sorted_g_pr + sorted_g_qr

    return g_r, g_rr, g_pr, g_qr # sorted_g_rr, sorted_g_pr, sorted_g_qr


def main(input_file='edges', num_elements=50, delimiter=';', over=True):

    global filename, overwrite, log, pagerank, node_list, elements

    log = 'Logs/log_{0}.txt'.format(input_file)
    filename = input_file
    overwrite = over
    print_log('starting...')
    start_time = time.time()
    graph = file_to_graph(delimiter)
    print_log('calculating the pagerank values...')
    pagerank = nx.pagerank(graph)
    try:
        with open('Datasets/node_list_{0}'.format(filename), 'rb') as fp:
            node_list = pickle.load(fp)
    except:
        node_list = list(graph)
    elements = node_list[:num_elements]
    sort_elements()
    # print_log('a: \n', a)

    # Creating the graph is necessary for the first time only, can be read from the files otherwise
    g = get_google_matrix(graph)

    size = len(graph)
    print_log('Starting the code for a square matrix of size {0} and count of {1}'.format(size, num_elements))
    g_r, g_rr, g_pr, g_qr = calc_g_r(g, num_elements)
    print_log('saving the results to files')
    np.save('Datasets/results/g_rr_{0}.npy'.format(filename), g_rr)
    np.save('Datasets/results/g_pr_{0}.npy'.format(filename), g_pr)
    np.save('Datasets/results/g_qr_{0}.npy'.format(filename), g_qr)
    np.save('Datasets/results/g_r_{0}.npy'.format(filename), g_r)
    plt.matshow(g_qr)
    plt.title('g_qr')
    plt.colorbar()
    plt.savefig('Results/g_qr_{0}.png'.format(filename))
    plt.matshow(g_pr)
    plt.title('g_pr')
    plt.colorbar()
    plt.savefig('Results/g_pr_{0}.png'.format(filename))
    plt.matshow(g_rr)
    plt.title('g_rr')
    plt.colorbar()
    plt.savefig('Results/g_rr_{0}.png'.format(filename))
    plt.matshow(g_r)
    plt.title('g_r')
    plt.colorbar()
    plt.savefig('Results/g_r_{0}.png'.format(filename))
    print_log('finished calculating g_r, g_rr, g_pr, and g_qr for the file {0}.'.format(filename))
    print_log('it took {0} seconds to run the program'.format(time.time() - start_time))


if __name__ == '__main__':
    for filename in os.listdir('Datasets/others/'):
        print(filename, filename[:filename.find('_')])
        # main(input_file=filename, num_elements=int(filename[:filename.find('_')]), delimiter=';')
        main(input_file=filename, num_elements=10, delimiter=';')
