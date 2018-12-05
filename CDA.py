import torch
import numpy as np
import torch.nn.functional as F

def pseudo_select(target_pred, num_classes, mode = 'global'):
    """
    perform pseudo label selection with information entropy
    :param target_pred: probability classification result, n * c
    :param num_classes: c, the number of total class
    :return: selected indexes
    """


    target_pred_np = np.array(F.softmax(target_pred, dim=1).data)
    batch_size = target_pred_np.shape[0]

    entropy = -(np.nan_to_num(np.log(target_pred_np)) * target_pred_np).sum(axis=1)
    if mode == 'global':
        mean_entropy = entropy.mean()
        selected_idx = entropy < mean_entropy
    else:
        selected_idx = np.full(batch_size, True)
        for iClass in range(0, num_classes):
            class_idx = target_pred_np == iClass
            if len(class_idx) == 0:
                continue
            mean_entropy = entropy[class_idx].mean()
            selected_idx[selected_idx == iClass] = entropy[class_idx] < mean_entropy

    return np.array(range(0, batch_size))[selected_idx]

def get_domain_class(labels, domains, iClass, domain):
    return [i for i in range(0, len(labels)) if (labels[i] == iClass and domains[i] == domain)]

def get_domain(domains, domain):
    #print(domains)
    return [i for i in range(0, len(domains)) if (domains[i] == domain)]


def mmd_cond(source, source_label, source_domain, target, target_label,target_domain, num_classes):
    loss = 0
    for iClass in range(0, num_classes):
        A_idx_s = get_domain_class(source_label, source_domain, iClass, 0)
        A_idx_t = get_domain_class(target_label, target_domain, iClass, 0)
        B_idx_s = get_domain_class(source_label, source_domain, iClass, 1)
        B_idx_t = get_domain_class(target_label, target_domain, iClass, 1)

        A_total = torch.cat( (source[A_idx_s], target[A_idx_t]) )
        B_total = torch.cat( (source[B_idx_s], target[B_idx_t]) )
        if len(A_total) == 0 or len(B_total) == 0:
            continue
        loss += (A_total.mean(dim=0) - B_total.mean(dim=0)).norm()

    return loss

def mmd_loss(source, source_domain, target, target_domain, acc_rbf = 1):
    loss = 0

    A_idx_s = get_domain(source_domain, 0)
    A_idx_t = get_domain(target_domain, 0)
    B_idx_s = get_domain(source_domain, 1)
    B_idx_t = get_domain(target_domain, 1)

    A_total = torch.cat( (source[A_idx_s], target[A_idx_t]) )
    B_total = torch.cat( (source[B_idx_s], target[B_idx_t]) )

    if len(A_total) == 0 or len(B_total) == 0:
        print('A_total or B_total empty!')
        return 0


    loss = mmd_rbf_noaccelerate(A_total, B_total)

    return loss

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    #print(loss)
    return loss

