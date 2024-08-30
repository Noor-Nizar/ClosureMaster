from torch import nn

mse_loss = nn.MSELoss()

def WMSELoss(pred, gt, recon_weight=0.75):
    
    img_pred, img_gt = pred[:, :3], gt[:, :3]
    seg_pred, seg_gt  = pred[:, 3:], gt[:, 3:]

    recon_loss = mse_loss(img_pred, img_gt)
    seg_loss = mse_loss(seg_pred, seg_gt)

    return recon_weight*recon_loss + (1-recon_weight)*seg_loss