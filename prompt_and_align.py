import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from Process.lm_loadsplits import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='Prompt-and-Align', type=str)
parser.add_argument('--iters', default=20, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--n_epochs', default=3, type=int)
parser.add_argument('--n_samples', default=16, type=int)
parser.add_argument('--u_thres', default=5, type=int)
args = parser.parse_args()
device = torch.device("cuda")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)

datasetname=args.dataset_name
batch_size = args.batch_size
user_threshold = args.u_thres
n_samples = args.n_samples
max_len = 512
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_conf, adj = pickle.load(open('data/adjs/user_t' + str(user_threshold) + '/' + datasetname + '_nn_relations_' + str(n_samples) + '.pkl', 'rb'))
A_nn = adj.todense()
A_nn = torch.FloatTensor(A_nn).to(device)
train_conf = torch.Tensor(train_conf).to(device)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        text = f"{tokenizer.mask_token}: " + text

        #softmax across all vocab + bce
        label_logit = self.labels[item]
        if label_logit == 0:
            label = np.array([1, 0])
        elif label_logit == 1:
            label = np.array([0, 1])

        encoding = self.tokenizer.encode_plus(text, add_special_tokens = True, max_length = self.max_len,
                pad_to_max_length = True, truncation = True, return_token_type_ids = False, return_attention_mask = True, return_tensors = 'pt')
        token_ids = encoding['input_ids']
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

        return {
            'news_text': text,
            'input_ids': token_ids.flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label),
            'masked_pos': torch.tensor(masked_position, dtype=torch.long),
        }

    def __len__(self):
        return len(self.texts)



class BERTPrompt(nn.Module):
    def __init__(self):
        super(BERTPrompt, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, masked_position):

        input_ids = input_ids.squeeze(1)
        output = self.bert(input_ids=input_ids)
        last_hidden_state = output[0]
        mask_hidden_state = last_hidden_state.select(1, masked_position)
        mask_hidden_state = self.softmax(mask_hidden_state)
        real_id = tokenizer.convert_tokens_to_ids('news')
        fake_id = tokenizer.convert_tokens_to_ids('rumor')
        indices = torch.tensor([real_id, fake_id]).to(device)
        probs = mask_hidden_state.index_select(1, indices)


        return probs


def create_train_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts = contents, labels = np.array(labels), tokenizer = tokenizer, max_len = max_len)
    
    return DataLoader(ds, batch_size = batch_size, shuffle = True, num_workers = 5)

def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts = contents, labels = np.array(labels), tokenizer = tokenizer, max_len = max_len)
    
    return DataLoader(ds, batch_size = batch_size, shuffle = False, num_workers = 0)



def train_model(args, x_train, x_test, y_train, y_test, tokenizer, max_len, n_epochs, batch_size, datasetname, iter):

    model = BERTPrompt().to(device)
    optimizer = AdamW(model.parameters(), lr = 5e-5)
    train_loader = create_train_loader(x_train, y_train, tokenizer, max_len, batch_size)
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)

    total_steps = len(train_loader) * n_epochs

    if len(x_train) == 16:
        total_steps = 5 * n_epochs
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    for epoch in range(n_epochs):

        model.train()
        avg_loss = []
        avg_acc = []
        for Batch_data in tqdm(train_loader):

            input_ids = Batch_data["input_ids"].to(device)
            targets = Batch_data["labels"].to(device)
            target_logits = torch.nonzero(targets)[:,-1]

            out_labels = model(input_ids = input_ids, masked_position = Batch_data["masked_pos"][0].item())

            loss_func = nn.BCELoss()
            loss = loss_func(out_labels,targets)

            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
            _, pred = out_labels.max(dim = -1)
            correct = pred.eq(target_logits).sum().item()
            train_acc = correct / len(targets)
            avg_acc.append(train_acc)
        
        train_loss = np.mean(avg_loss)
        train_acc = np.mean(avg_acc)

        print("Iter {:03d} | Epoch {:05d} | Train Acc. {:.4f} | Train_Loss {:.4f}".format(iter, epoch, train_acc, train_loss))


        model.eval()
        y_pred = []
        y_test = []
        y_probs = []

        for Batch_data in tqdm(test_loader):
            
            with torch.no_grad():
                input_ids = Batch_data["input_ids"].to(device)
                targets = Batch_data["labels"].to(device)
                target_logits = torch.nonzero(targets)[:,-1]
               
                test_out = model(input_ids = input_ids, masked_position = Batch_data["masked_pos"][0].item())
                _, val_pred = test_out.max(dim = -1)

                y_pred.append(val_pred)
                y_test.append(target_logits)
                y_probs.append(test_out)

        y_pred = torch.cat(y_pred, dim = 0)
        y_test = torch.cat(y_test, dim = 0)
        y_probs = torch.cat(y_probs, dim = 0)
        y_probs = F.softmax(y_probs, dim = -1)
        
        diff = []
        for i in range(y_probs.shape[0]):
            diff.append(abs(y_probs[i][0] - y_probs[i][1]).item())
        
        threshold = np.percentile(np.array(diff), 95)
        
        for i in range(y_probs.shape[0]):
            if y_probs[i][0] - y_probs[i][1] >= threshold:
                y_probs[i][0] = 1.
                y_probs[i][1] = 0.
            if y_probs[i][1] - y_probs[i][0] >= threshold:
                y_probs[i][0] = 0.
                y_probs[i][1] = 1.
        y_probs = torch.cat([train_conf, y_probs], dim = 0)                      

        
        correct = y_pred.eq(y_test).sum().item()
        test_acc = correct / len(y_test)

        print("Iter {:03d} | Epoch {:05d} | Prompting Test Acc. {:.4f}".format(iter, epoch, test_acc))

        if epoch == n_epochs - 1:

            print("propagation on news-news graph")

            y_probs_upd = torch.matmul(A_nn, y_probs)

            y_probs_upd = torch.matmul(A_nn, y_probs_upd)
            _, y_pred_upd = y_probs_upd.max(dim=1)
            y_pred_upd = y_pred_upd[n_samples:]
            
            acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred_upd.detach().cpu().numpy())
            precision, recall, fscore, _ = score(y_test.detach().cpu().numpy(), y_pred_upd.detach().cpu().numpy(), average='macro')
            

    print("-----------------End of Iter {:03d}-----------------".format(iter))
    print(['Final Test Accuracy:{:.4f}'.format(acc),
        'Precision:{:.4f}'.format(precision),
        'Recall:{:.4f}'.format(recall),
        'F1:{:.4f}'.format(fscore)])
    
    return acc, precision, recall, fscore


n_epochs = args.n_epochs
batchsize = args.batch_size
iterations = args.iters
test_accs = []
prec_all, rec_all, f1_all = [], [], []

# value of args.n_samples in [16, 32, 64, 128]
x_train, x_test, y_train, y_test = get_splits_fewshot(datasetname, args.n_samples)


for iter in range(iterations):
    acc, prec, recall, f1 = train_model(args,
                                        x_train, x_test, y_train, y_test,
                                        tokenizer,
                                        max_len,
                                        n_epochs,
                                        batch_size,
                                        datasetname,
                                        iter)

    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(recall)
    f1_all.append(f1)

print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs) / iterations, sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))


with open('logs/log_' +  datasetname + '_fewshot_' + str(args.n_samples) + '_samples_' + args.model_name + '_t' + str(user_threshold) + '.' + 'iter' + str(iterations), 'a+') as f:
    f.write('Prompt template: [MASK]: <text>')
    f.write('\n')
    f.write('Label words: news, rumor')
    f.write('\n')
    f.write('All acc.s:{}\n'.format(test_accs))
    f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))
    f.write('\n')
