from torch.utils.data import DataLoader
from mdd_dataset import MDDDataset
from transformers import AutoTokenizer
from model.electra import ElectraForSequenceClassification
import torch
from utils.utils import *
from torch import nn, optim
from models.pdmc import *
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score
import wandb
from sklearn.metrics import precision_score, recall_score
# from accelerate import Accelerator
# accelerator = Accelerator()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["daic", "edaic"], default='daic')
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")
parser.add_argument("--best_acc", type=float, default=0.1)
parser.add_argument("--wandb_name", type=str, default='none')
parser.add_argument("--unimodal", type=str, default='text')
parser.add_argument("--pretrain_path", type=str, default='/data/CMU/pretrain/google/electra-base-discriminator')
parser.add_argument("--train_meta_path", type=str, default='./daic/train_split_Depression_AVEC2017.csv')
parser.add_argument("--valid_meta_path", type=str, default='./daic/dev_split_Depression_AVEC2017.csv')
parser.add_argument("--test_meta_path", type=str, default='./daic/test_split_Depression_AVEC2017.csv')
parser.add_argument("--classifier_dropout", type=float, default=0.1)
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--index", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create dataloader
def create_dataloader(json_file, batch_size=32, shuffle=True, max_length=400, tokenizer=None):
    dataset = MDDDataset(json_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
def get_dataloader():
    bert_tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)
    train_dataloader = create_dataloader(json_file=args.train_meta_path, batch_size=args.train_batch_size, tokenizer=bert_tokenizer)
    valid_dataloader = create_dataloader(json_file=args.valid_meta_path,  batch_size=args.dev_batch_size, tokenizer=bert_tokenizer, shuffle=False)
    test_dataloader = create_dataloader(json_file=args.test_meta_path,  batch_size=args.test_batch_size, tokenizer=bert_tokenizer, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader


def prepare_training(train_dataloader):
    # model = ElectraForSequenceClassification.from_pretrained("bhadresh-savani/electra-base-emotion", num_labels=num_labels)
    # model = ElectraForSequenceClassification.from_pretrained(args.pretrain_path, num_labels=args.num_labels)
    model = TopicModel(args)
    # model = DecisionFusionModel(args)
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * args.epochs)
    return model, optimizer, scheduler

def train_epoch(model, train_dataloader, optimizer, scheduler, epoch=0):
    step = 0
    tr_loss = 0
    preds = []
    y_test = []
    model.train()
    for step, (meta, batch) in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = [item.to(device) for item in batch]
        family_input_ids, family_attention_mask, work_input_ids, work_attention_mask, mental_input_ids, mental_attention_mask, medical_input_ids, medical_attention_mask, abstract_input_ids, abstract_attention_mask, label_id = batch
        step += 1
        optimizer.zero_grad()
        outputs = model(
            family_input_ids, family_attention_mask, 
            work_input_ids, work_attention_mask, 
            mental_input_ids, mental_attention_mask, 
            medical_input_ids, medical_attention_mask, 
            abstract_input_ids, abstract_attention_mask,
            )
        loss_fct = CrossEntropyLoss()
        
        logits = outputs
        loss = loss_fct(logits, label_id.view(-1))
        total_loss = loss 
        total_loss.backward()
        # accelerator.backward(loss)
        tr_loss += total_loss.item()
        optimizer.step()
        scheduler.step()

        _, predicted = torch.max(logits.data, dim=1)
        y_test_val = label_id.view(-1).cpu().detach().tolist()
        preds_val = predicted.cpu().detach().tolist()
        y_test += y_test_val
        preds += preds_val
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    tr_loss /= step
    return tr_loss, acc, f1

def test_epoch(model: nn.Module, test_dataloader: DataLoader, domain=0, epoch=-1, test='test'):
    model.eval()
    preds = []
    y_test = []
    metas = []
    with torch.no_grad():
        for step, (meta, batch) in enumerate(tqdm(test_dataloader, desc="Iteration")):
            batch = [item.to(device) for item in batch]
            family_input_ids, family_attention_mask, work_input_ids, work_attention_mask, mental_input_ids, mental_attention_mask, medical_input_ids, medical_attention_mask, abstract_input_ids, abstract_attention_mask, label_id = batch
            step += 1
            outputs = model(
                family_input_ids, family_attention_mask, 
                work_input_ids, work_attention_mask, 
                mental_input_ids, mental_attention_mask, 
                medical_input_ids, medical_attention_mask, 
                abstract_input_ids, abstract_attention_mask
                )
            logits = outputs 
            _, predicted = torch.max(logits.data, dim=1)
            y_test_val = label_id.view(-1).cpu().detach().tolist()
            preds_val = predicted.cpu().detach().tolist()
            y_test += y_test_val
            preds += preds_val
            metas += meta.cpu().detach().tolist()

        print(f'{test} meta  :', meta)
        print(f'{test} y_test:', y_test)
        print(f'{test} preds :', preds)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average='macro')
        f1_micro = f1_score(y_test, preds, average='micro')
        f1_weight = f1_score(y_test, preds, average='weighted')

        p_macro = precision_score(y_test, preds, average='macro')  
        p_micro = precision_score(y_test, preds, average='micro')  
        p_weight = precision_score(y_test, preds, average='weighted')  

        r_macro = recall_score(y_test, preds, average='macro')
        r_micro = recall_score(y_test, preds, average='micro')
        r_weight = recall_score(y_test, preds, average='weighted')

    return acc, f1_macro, f1_micro, f1_weight, p_macro, p_micro, p_weight, r_macro, r_micro, r_weight


def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
):
    valid_accs, valid_f1s = [], []
    test_accs, test_f1s = [], []
    valid_precisions, valid_recalls = [], []
    test_precisions, test_recalls = [], []
    for epoch_i in range(int(args.epochs)):
        train_loss, train_acc, train_f1 = train_epoch(model, train_dataloader, optimizer, scheduler, epoch=epoch_i)
        val_acc, val_f1_macro, val_f1_micro, val_f1_weight, val_p_macro, val_p_micro, val_p_weight, val_r_macro, val_r_micro, val_r_weight = test_epoch(model, dev_dataloader, optimizer, test='valid', epoch=epoch_i)
        test_acc, test_f1_macro, test_f1_micro, test_f1_weight, test_p_macro, test_p_micro, test_p_weight, test_r_macro, test_r_micro, test_r_weight = test_epoch(model, test_dataloader, test='test', epoch=epoch_i)
        print(
            f"epoch:{epoch_i}, train_loss:{train_loss}, valid_acc:{val_acc}, valid_f1:{val_f1_weight}, test_acc:{test_acc}, test_f1:{test_f1_weight}, test_precision:{test_p_weight}, test_recall:{test_r_weight}"
        )

        valid_accs.append(val_acc)
        valid_f1s.append(val_f1_weight)
        valid_precisions.append(val_p_weight)
        valid_recalls.append(val_r_weight)
        test_accs.append(test_acc)
        test_f1s.append(test_f1_weight)
        test_precisions.append(test_p_weight)
        test_recalls.append(test_r_weight)

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
            
                    "valid_acc": val_acc,
                    "valid_f1": val_f1_weight,
                    "valid_precision": val_p_weight,
                    "valid_recall": val_r_weight,
        
                    "best_valid_acc": max(valid_accs),
                    "best_valid_f1": max(valid_f1s),
                    "best_valid_precision": max(valid_precisions),
                    "best_valid_recalls": max(valid_recalls),

                    "test_acc": test_acc,
                    "test_f1": test_f1_weight,
                    "test_precision": test_p_weight,
                    "test_recall": test_r_weight,
            
                    "best_test_acc": max(test_accs),
                    "best_test_f1": max(test_f1s),
                    "best_test_precisions": max(test_precisions),
                    "best_test_recalls": max(test_recalls),


                    "val_f1_macro": val_f1_macro,
                    "val_f1_micro": val_f1_micro,
                    "val_p_macro": val_p_macro,
                    "val_p_micro": val_p_micro,
                    "val_r_macro": val_r_macro,
                    "val_r_micro": val_r_micro,


                    "test_f1_macro": test_f1_macro,
                    "test_f1_micro": test_f1_micro,
                    "test_p_macro": test_p_macro,
                    "test_p_micro": test_p_micro,
                    "test_r_macro": test_r_macro,
                    "test_r_micro": test_r_micro,
                }
            )
        )



def main():
    wandb.init(project="MDD-Anchor", name=args.wandb_name)
    wandb.config.update(args)
    args.seed = 7206
    set_random_seed(args.seed)
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader()
    model, optimizer, scheduler = prepare_training(train_dataloader)
    train(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
        )
if __name__ == "__main__":
    main()

