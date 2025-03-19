class MutualRestraintLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MutualRestraintLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, targets):
        # 分离正常和疾病的预测
        normal_pred = outputs[:, 0]
        disease_pred = outputs[:, 1:]
        
        # 分离正常和疾病的标签
        normal_target = targets[:, 0]
        disease_target = targets[:, 1:]
        
        # 基础分类损失
        normal_loss = self.bce(normal_pred, normal_target)
        disease_loss = self.bce(disease_pred, disease_target)
        
        # 互斥损失
        mutual_loss = torch.mean(normal_pred * disease_pred.mean(dim=1))
        
        # 组合损失
        total_loss = normal_loss + disease_loss + self.alpha * mutual_loss
        
        return total_loss