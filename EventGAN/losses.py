import torch
import torch.nn.functional as F
import numpy as np

def resize_like(image, target):
    return torch.nn.functional.interpolate(image,
                                           size=target.shape[2:],
                                           mode='bilinear',
                                           align_corners=True)
    
def apply_flow(prev, flow, ignore_ooi=False):
    """ Warp prev to cur through flow
    I_c(x,y) = I_p(x+f_u(x,y), y+f_v(x,y))
    """
    batch_size, _, height, width = prev.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(prev)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(prev)

    x_shift = flow[:, 0, :, :]# / flow.shape[-1]
    y_shift = flow[:, 1, :, :]# / flow.shape[-1]
    flow_field = torch.stack((x_base+x_shift, y_base+y_shift), dim=3)

    output = F.grid_sample(prev, 2 * flow_field - 1, mode='bilinear',
                           padding_mode='zeros')
    
    if ignore_ooi:
        return output, output.new_ones(output.shape).byte()
    else:
        mask = (flow_field[...,0]>-1.0) * (flow_field[...,0]<1.0)\
             * (flow_field[...,1]>-1.0) * (flow_field[...,1]<1.0)

        return output, mask[:,None,:,:]

def gradient(I):
    """
    Arguments:
    - I - shape N1,...,Nn,C,H,W
    Returns:
    - dx - shape N1,...,Nn,C,H,W
    - dy - shape N1,...,Nn,C,H,W
    """

    dy = I.new_zeros(I.shape)
    dx = I.new_zeros(I.shape)

    dy[...,1:,:] = I[...,1:,:] - I[...,:-1,:]
    dx[...,:,1:] = I[...,:,1:] - I[...,:,:-1]

    return dx, dy
    
def flow_smoothness(flow, mask=None):
    dx, dy = gradient(flow)
    
    if mask is not None:
        mask = mask.expand(-1,2,-1,-1)
        loss = (charbonnier_loss(dx[mask]) \
                + charbonnier_loss(dy[mask])) / 2.
    else:
        loss = (charbonnier_loss(dx) \
                + charbonnier_loss(dy)) / 2.
        
    return loss

def charbonnier_loss(error, alpha=0.45, mask=None):
    charbonnier = (error ** 2. + 1e-5 ** 2.) ** alpha
    if mask is not None:
        mask = mask.float()
        loss = torch.mean(torch.sum(charbonnier * mask, dim=(1, 2, 3)) / \
                          torch.sum(mask, dim=(1, 2, 3)))
    else:
        loss = torch.mean(charbonnier)
    return loss
    
def squared_error(input, target):
    return torch.sum((input - target) ** 2)

def generator_loss(loss_func, fake):
    if loss_func == "wgan":
        return -fake.mean()
    elif loss_func == "gan":
        return F.binary_cross_entropy_with_logits(input=fake, target=torch.ones_like(fake).cuda())
    elif loss_func == "lsgan":
        return squared_error(input=fake, target=torch.ones_like(fake).cuda()).mean() 
    elif loss_func == "hinge":
        return -fake.mean()
    else:
        raise Exception("Invalid loss_function")

def discriminator_loss(loss_func, real, fake):
    if loss_func == "wgan":
        real_loss = -real.mean()
        fake_loss = fake.mean()
    elif loss_func == "gan":
        real_loss = F.binary_cross_entropy_with_logits(input=real,
                                                       target=torch.ones_like(real).cuda())
        fake_loss = F.binary_cross_entropy_with_logits(input=fake,
                                                       target=torch.zeros_like(fake).cuda())
    elif loss_func == "lsgan":
        real_loss = squared_error(input=real, target=torch.ones_like(real).cuda()).mean()
        fake_loss = squared_error(input=fake, target=torch.zeros_like(fake).cuda()).mean()
    elif loss_func == "hinge":
        real_loss = F.relu(1.0 - real).mean()
        fake_loss = F.relu(1.0 + fake).mean()
    else:
        raise Exception("Invalid loss_function")

    return real_loss + fake_loss

def multi_scale_flow_loss(prev_image, next_image, flow_list, valid_mask):
    # Multi-scale loss
    total_photo_loss = 0.
    total_smooth_loss = 0.
    pred_image_list = []

    for i, flow in enumerate(flow_list):
        # upsample the flow
        up_flow = F.interpolate(flow, size=(prev_image.shape[2], prev_image.shape[3]),
                                mode='nearest')
        # apply the flow to the current image
        pred_image, mask = apply_flow(prev_image, up_flow)

        scale = 2. ** (i-len(flow_list)+1)
        total_photo_loss += charbonnier_loss(pred_image - next_image, mask=mask*valid_mask) * scale
        total_smooth_loss += flow_smoothness(flow) * scale
            
        pred_image_list.append(pred_image)
        
    return total_photo_loss, total_smooth_loss, pred_image_list
