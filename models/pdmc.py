class TopicModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert_model = ElectraModel.from_pretrained(args.pretrain_path)

        self.family_pos = nn.Parameter(torch.from_numpy(np.load('anchor/family_pos.npy')).to('cuda'))
        self.family_neg = nn.Parameter(torch.from_numpy(np.load('anchor/family_neg.npy')).to('cuda'))
        self.family_neu = nn.Parameter(torch.from_numpy(np.load('anchor/family_neu.npy')).to('cuda'))

        self.work_pos = nn.Parameter(torch.from_numpy(np.load('anchor/work_pos.npy')).to('cuda'))
        self.work_neg = nn.Parameter(torch.from_numpy(np.load('anchor/work_neg.npy')).to('cuda'))
        self.work_neu = nn.Parameter(torch.from_numpy(np.load('anchor/work_neu.npy')).to('cuda'))

        self.mental_pos = nn.Parameter(torch.from_numpy(np.load('anchor/mental_pos.npy')).to('cuda'))
        self.mental_neg = nn.Parameter(torch.from_numpy(np.load('anchor/mental_neg.npy')).to('cuda'))
        self.mental_neu = nn.Parameter(torch.from_numpy(np.load('anchor/mental_neu.npy')).to('cuda'))

        self.medical_pos = nn.Parameter(torch.from_numpy(np.load('anchor/medical_pos.npy')).to('cuda'))
        self.medical_neg = nn.Parameter(torch.from_numpy(np.load('anchor/medical_neg.npy')).to('cuda'))
        self.medical_neu = nn.Parameter(torch.from_numpy(np.load('anchor/medical_neu.npy')).to('cuda'))


        self.abstract_pos = nn.Parameter(torch.from_numpy(np.load('anchor/abstract_pos.npy')).to('cuda'))
        self.abstract_neg = nn.Parameter(torch.from_numpy(np.load('anchor/abstract_neg.npy')).to('cuda'))
        self.abstract_neu = nn.Parameter(torch.from_numpy(np.load('anchor/abstract_neu.npy')).to('cuda'))

        self.family_anchor = torch.cat([self.family_pos, self.family_neu, self.family_neg])
        self.work_anchor = torch.cat([self.work_pos, self.work_neu, self.work_neg])
        self.mental_anchor = torch.cat([self.mental_pos, self.mental_neu, self.mental_neg])
        self.medical_anchor = torch.cat([self.medical_pos, self.medical_neu, self.medical_neg])
        self.abstract_anchor = torch.cat([self.abstract_pos, self.abstract_neu, self.abstract_neg])

        # classifier    
        self.family_classifier = ClassificationHead(args)
        self.work_classifier = ClassificationHead(args)
        self.mental_classifier = ClassificationHead(args)
        self.medical_classifier = ClassificationHead(args)
        self.abstract_classifier = ClassificationHead(args)
        
        self.softmax = nn.Softmax(dim=1)
        self.index = args.index
        self.pred_w = nn.Parameter(torch.randn(5))
        self.reset_parameter()
    def reset_parameter(self):
        self.pred_w.data.fill_(1.0)

    def forward(self, family_input_ids, family_attention_mask, work_input_ids, work_attention_mask, mental_input_ids, mental_attention_mask, medical_input_ids, medical_attention_mask, abstract_input_ids, abstract_attention_mask):
        family_feats = self.bert_model(family_input_ids, family_attention_mask)[0]
        work_feats = self.bert_model(work_input_ids, work_attention_mask)[0]
        mental_feats = self.bert_model(mental_input_ids, mental_attention_mask)[0]
        medical_feats = self.bert_model(medical_input_ids, medical_attention_mask)[0]
        abstract_feats = self.bert_model(abstract_input_ids, abstract_attention_mask)[0]
        # anchor similarity



        family_similarity = torch.mm(family_feats[:, 0, :], self.family_anchor.transpose(0, 1))
        work_similarity = torch.mm(work_feats[:, 0, :], self.work_anchor.transpose(0, 1))
        mental_similarity = torch.mm(mental_feats[:, 0, :], self.mental_anchor.transpose(0, 1))
        medical_similarity = torch.mm(medical_feats[:, 0, :], self.medical_anchor.transpose(0, 1))
        abstract_similarity = torch.mm(abstract_feats[:, 0, :], self.abstract_anchor.transpose(0, 1))
        # norm_family_feats = torch.norm(family_feats[:, 0, :], dim=1)
        # norm_family_anchor = torch.norm(self.family_anchor, dim=1)
        # norm_family_feats = torch.norm(family_feats[:, 0, :], dim=1)
        # norm_family_anchor = torch.norm(self.family_anchor, dim=1)
        # norm_family_feats = torch.norm(family_feats[:, 0, :], dim=1)
        # norm_family_anchor = torch.norm(self.family_anchor, dim=1)
        # norm_family_feats = torch.norm(family_feats[:, 0, :], dim=1)
        # norm_family_anchor = torch.norm(self.family_anchor, dim=1)

        family_prob = F.softmax(family_similarity, dim=1)
        work_prob = F.softmax(work_similarity, dim=1)
        mental_prob = F.softmax(mental_similarity, dim=1)
        medical_prob = F.softmax(medical_similarity, dim=1)
        abstract_prob = F.softmax(abstract_similarity, dim=1)
        # select anchor
        family_max_indices = torch.argmax(family_prob, dim=1)
        family_selected_anchor = self.family_anchor[family_max_indices]
        work_max_indices = torch.argmax(work_prob, dim=1)
        work_selected_anchor = self.work_anchor[work_max_indices]
        mental_max_indices = torch.argmax(mental_prob, dim=1)
        mental_selected_anchor = self.mental_anchor[mental_max_indices]
        medical_max_indices = torch.argmax(medical_prob, dim=1)
        medical_selected_anchor = self.medical_anchor[medical_max_indices]
        abstract_max_indices = torch.argmax(abstract_prob, dim=1)
        abstract_selected_anchor = self.abstract_anchor[abstract_max_indices]
        # weight merge 
        family_weights = torch.matmul(family_feats, family_selected_anchor.unsqueeze(-1))  # shape: [N, k, 1]
        family_feats = family_feats * family_weights  # shape: [N, k, D]
        work_weights = torch.matmul(work_feats, work_selected_anchor.unsqueeze(-1))  # shape: [N, k, 1]
        work_feats = work_feats * work_weights  # shape: [N, k, D]
        mental_weights = torch.matmul(mental_feats, mental_selected_anchor.unsqueeze(-1))  # shape: [N, k, 1]
        mental_feats = mental_feats * mental_weights  # shape: [N, k, D]
        medical_weights = torch.matmul(medical_feats, medical_selected_anchor.unsqueeze(-1))  # shape: [N, k, 1]
        medical_feats = medical_feats * medical_weights  # shape: [N, k, D]
        abstract_weights = torch.matmul(abstract_feats, abstract_selected_anchor.unsqueeze(-1))  # shape: [N, k, 1]
        abstract_feats = abstract_feats * abstract_weights  # shape: [N, k, D]

        family_feats = torch.sum(family_feats, dim=1)
        work_feats = torch.sum(work_feats, dim=1)
        mental_feats = torch.sum(mental_feats, dim=1)
        medical_feats = torch.sum(medical_feats, dim=1)
        abstract_feats = torch.sum(abstract_feats, dim=1)

        family_logits = self.family_classifier(family_feats)
        work_logits = self.work_classifier(work_feats)
        mental_logits = self.mental_classifier(mental_feats)
        medical_logits = self.medical_classifier(medical_feats)
        abstract_logits = self.abstract_classifier(medical_feats)

        family_logits = family_logits.unsqueeze(1)
        work_logits = work_logits.unsqueeze(1)
        mental_logits = mental_logits.unsqueeze(1)
        medical_logits = medical_logits.unsqueeze(1)
        abstract_logits = abstract_logits.unsqueeze(1)

        x = torch.cat([family_logits, work_logits, mental_logits, medical_logits, abstract_logits], dim=1)
        x = torch.logsumexp(x, dim=1)
        return x