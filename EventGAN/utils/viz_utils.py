import torch
import numpy as np
import matplotlib.colors as colors

def single_flow2rgb(flow_x, flow_y, hsv_buffer=None):
    if hsv_buffer is None:
        hsv_buffer = np.empty((flow_x.shape[0],flow_x.shape[1],3))
    hsv_buffer[:,:,1] = 1.0
    hsv_buffer[:,:,0] = (np.arctan2(flow_y,flow_x)+np.pi)/(2.0*np.pi)

    hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )

    flat = hsv_buffer[:,:,2].reshape((-1))
    m = np.nanmax(flat[np.isfinite(flat)])
    if not np.isclose(m,0.0):
        hsv_buffer[:,:,2] /= m

    return colors.hsv_to_rgb(hsv_buffer)

def flow2rgb(flow, squeeze=True):
    flow_x, flow_y = flow[:,0,:,:].cpu().detach().numpy(), flow[:,1,:,:].cpu().detach().numpy()

    if squeeze:
        flow_x = flow_x.squeeze()
        flow_y = flow_y.squeeze()

    hsv_buffer = np.empty((flow_x.shape[0],flow_x.shape[1],flow_x.shape[2],3))

    for i in range(flow_x.shape[0]):
        single_flow2rgb(flow_x[i,...], flow_y[i,...], hsv_buffer[i,...])

    return torch.from_numpy(colors.hsv_to_rgb(hsv_buffer).transpose((0,3,1,2)))

def normalize_event_image(event_image, clamp_val=2., normalize_events=True):
    if not normalize_events:
        return event_image
    else:
        return torch.clamp(event_image, 0, clamp_val) / clamp_val# + 1.) / 2.

def gen_event_images(event_volume, prefix, device="cuda", clamp_val=2., normalize_events=True):
    n_bins = int(event_volume.shape[1] / 2)
    time_range = torch.tensor(np.linspace(0.1, 1, n_bins), dtype=torch.float32).to(device)
    time_range = torch.reshape(time_range, (1, n_bins, 1, 1))
    
    pos_event_image = torch.sum(
        event_volume[:, :n_bins, ...] * time_range / \
        (torch.sum(event_volume[:, :n_bins, ...], dim=1, keepdim=True) + 1e-5),
        dim=1, keepdim=True)
    neg_event_image = torch.sum(
        event_volume[:, n_bins:, ...] * time_range / \
        (torch.sum(event_volume[:, n_bins:, ...], dim=1, keepdim=True) + 1e-5),
        dim=1, keepdim=True)
    
    outputs = {
        '{}_event_time_image'.format(prefix) : (pos_event_image + neg_event_image) / 2.,
        '{}_event_image'.format(prefix) : normalize_event_image(
            torch.sum(event_volume, dim=1, keepdim=True)),
        '{}_event_image_x'.format(prefix) : normalize_event_image(
            torch.sum(event_volume.permute((0, 2, 1, 3)), dim=1, keepdim=True),
            normalize_events=normalize_events),
        '{}_event_image_y'.format(prefix) : normalize_event_image(
            torch.sum(event_volume.permute(0, 3, 1, 2), dim=1, keepdim=True),
            normalize_events=normalize_events)
    }
    
    return outputs
