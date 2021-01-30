import sys, os, pickle, time
import math, random
import numpy as np
import torch
from metric import evaluate
from utils import *
from utils_add import *


class FPMC():
    def __init__(self, n_item, n_factor, learn_rate, regular):

        self.item_set = set()
        self.n_item = n_item

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

    @staticmethod # 可以实例化类FPMC，然后实例化调用， 也可以不实例化，直接调用， eg： FPMC.dump()
    def dump(fpmcObj, fname):
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, 'rb'))

    def init_model(self, std=0.01):
        # self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        # self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        # self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    def compute_x(self, i, b_tm1):
        acc_val = 0.0
        for l in b_tm1:
            acc_val += np.dot(self.VIL[i], self.VLI[l])
        return (acc_val/len(b_tm1))

    def compute_x_batch(self, b_tm1):
        latter = np.mean(self.VIL_m_VLI[:, b_tm1], axis=1).T
        return latter

    def predict(self, data_list):
    
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        scores_result = []
        ground_truth = []

        for (i, b_tm1) in data_list:
            scores = self.compute_x_batch(b_tm1)
            scores_result.append(scores)
            ground_truth.append(i)
        scores_result = torch.tensor(scores_result)
        ground_truth = torch.tensor(ground_truth).long()
        return scores_result, ground_truth

    """
    def evaluation(self, data_list):
        # np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        correct_count = 0
        rr_list = []
        for (i, b_tm1) in data_list:
            scores = self.compute_x_batch(b_tm1)

            if i == scores.argmax():
                correct_count += 1

            rank = len(np.where(scores > scores[i])[0]) + 1
            rr = 1.0/rank
            rr_list.append(rr)

        try:
            acc = correct_count / len(rr_list)
            mrr = (sum(rr_list) / len(rr_list))
            return (acc, mrr)
        except:
            return (0.0, 0.0)
    """

    def learn_epoch(self, tr_data, neg_batch_size):
        for iter_idx in range(len(tr_data)):
            (i, b_tm1) = random.choice(tr_data)
            
            exclu_set = self.item_set - set([i])
            j_list = random.sample(exclu_set, neg_batch_size)
            
            z1 = self.compute_x(i, b_tm1)
            for j in j_list:

                z2 = self.compute_x(j, b_tm1)
                delta = 1 - sigmoid(z1 - z2)

                # VUI_update = self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])
                # VIUi_update = self.learn_rate * (delta * self.VUI[u] - self.regular * self.VIU[i])
                # VIUj_update = self.learn_rate * (-delta * self.VUI[u] - self.regular * self.VIU[j])

                # self.VUI[u] += VUI_update
                # self.VIU[i] += VIUi_update
                # self.VIU[j] += VIUj_update

                eta = np.mean(self.VLI[b_tm1], axis=0)
                VILi_update = self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                VILj_update = self.learn_rate * (-delta * eta - self.regular * self.VIL[j])
                VLI_update = self.learn_rate * ((delta * (self.VIL[i] - self.VIL[j]) / len(b_tm1)) - self.regular * self.VLI[b_tm1])

                self.VIL[i] += VILi_update
                self.VIL[j] += VILj_update
                self.VLI[b_tm1] += VLI_update
    

    def learnSBPR_FPMC(self,log_path_txt, tr_data, te_data=None, n_epoch=10, neg_batch_size=10, eval_per_epoch=True, Ks=[10,20,50], Ks_auc=[100]):
        
        best_result_recall = [0]*len(Ks)
        best_result_mrr = [0] * len(Ks)
        best_result_ndcg = [0] * len(Ks)
        best_result_auc = [0] * len(Ks_auc)

        best_epoch_auc = [0] * len(Ks_auc)
        best_epoch_ndcg = [0] * len(Ks)
        best_epoch_recall = [0]*len(Ks)
        best_epoch_mrr = [0]*len(Ks)
        
        for epoch in range(n_epoch):
            self.learn_epoch(tr_data, neg_batch_size=neg_batch_size)

            if eval_per_epoch == True:
                scores, ground_truth = self.predict(tr_data)
                recall, mrr, ndcg, auc = evaluate(scores=scores, ground_truth=ground_truth, Ks=Ks, Ks_auc=Ks_auc)

                # print train data result
                file_write(log_path_txt,f'Current Train Data Result Epoch {epoch}:')
                print_result(log_path_txt, 'Recall', recall,Ks)
                print_result(log_path_txt, 'MRR', mrr,Ks)
                print_result(log_path_txt, 'NDCG', ndcg, Ks)
                print_auc_result(log_path_txt, 'AUC', auc, Ks_auc)

            # start test
            scores, ground_truth = self.predict(te_data)
            recall, mrr, ndcg, auc = evaluate(scores=scores, ground_truth=ground_truth, Ks=Ks, Ks_auc=Ks_auc)

            for i, topk in enumerate(Ks):
                if recall[i] >= best_result_recall[i]:
                    best_result_recall[i] = recall[i]
                    best_epoch_recall[i] = epoch
                if mrr[i] >= best_result_mrr[i]:
                    best_result_mrr[i] = mrr[i]
                    best_epoch_mrr[i] = epoch
                if ndcg[i] >= best_result_ndcg[i]:
                    best_result_ndcg[i] = ndcg[i]
                    best_epoch_ndcg[i] = epoch
            for i, k in enumerate(Ks_auc):
                if auc[i] >= best_result_auc[i]:
                    best_result_auc[i] = auc[i]
                    best_epoch_auc[i] = epoch

            # print test result
            cprint(log_path_txt,f'Current Test Data Result Epoch {epoch}:')
            print_result(log_path_txt, 'Recall', recall,Ks)
            print_result(log_path_txt, 'MRR', mrr,Ks)
            print_result(log_path_txt, 'NDCG', ndcg, Ks)
            print_auc_result(log_path_txt, 'AUC', auc, Ks_auc)

            file_write(log_path_txt, 'Best Result: ')
            file_write(log_path_txt, f'best_epoch_recall: {best_epoch_recall}')
            file_write(log_path_txt, f'best_epoch_mrr: {best_epoch_mrr}')
            file_write(log_path_txt, f'best_epoch_ndcg: {best_epoch_ndcg}')
            file_write(log_path_txt, f'best_epoch_auc: {best_epoch_auc}')

            print_result(log_path_txt, 'Recall', best_result_recall, Ks)
            print_result(log_path_txt, 'MRR', best_result_mrr, Ks)
            print_result(log_path_txt, 'NDCG', best_result_ndcg, Ks)
            print_auc_result(log_path_txt, 'AUC', best_result_auc, Ks_auc)

        return None


if __name__ == "__main__":
    print("lala")