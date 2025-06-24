import numpy as np
import torch
from event_utils import *
from scipy.ndimage.filters import gaussian_filter
from abc import ABC, abstractmethod

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    Get a mask of the events that are within the given bounds
    """
    mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
    return mask

def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img

def interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2):
    """
    Accumulate x and y coords to an image using double weighted bilinear interpolation
    """
    for i in range(d_img.shape[0]):
        d_img[i].index_put_((pys,   pxs  ), w1[i] * (-(1.0-dys)) + w2[i] * (-(1.0-dxs)), accumulate=True)
        d_img[i].index_put_((pys,   pxs+1), w1[i] * (1.0-dys)    + w2[i] * (-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs  ), w1[i] * (-dys)       + w2[i] * (1.0-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs+1), w1[i] * dys          + w2[i] *  dxs, accumulate=True)

def events_to_image_drv(xn, yn, pn, jacobian_xn, jacobian_yn,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, compute_gradient=False):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    xt, yt, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(pn)
    xs, ys, ps, = xt.float(), yt.float(), pt.float()
    if compute_gradient:
        jacobian_x, jacobian_y = torch.from_numpy(jacobian_xn), torch.from_numpy(jacobian_yn)
        jacobian_x, jacobian_y = jacobian_x.float(), jacobian_y.float()
    if device is None:
        device = xs.device
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        zero_v = torch.tensor([0.])
        ones_v = torch.tensor([1.])
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs-pxs
    dys = ys-pys
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask
    img = torch.zeros(img_size).to(device)
    interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)

    if compute_gradient:
        d_img = torch.zeros((2, *img_size)).to(device)
        w1 = jacobian_x*masked_ps
        w2 = jacobian_y*masked_ps
        interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2)
        d_img = d_img.numpy()
    else:
        d_img = None
    return img.numpy(), d_img

def get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, compute_gradient=False, use_polarity=True):
    """
    Given a set of parameters, events and warp function, get the warped image and derivative image
    if required.
    """
    if not use_polarity:
        ps = np.abs(ps)
    xs, ys, jx, jy = warpfunc.warp(xs, ys, ts, ps, ts[-1], params, compute_grad=compute_gradient)
    mask = events_bounds_mask(xs, ys, 0, img_size[1], 0, img_size[0])
    xs, ys, ts, ps = xs*mask, ys*mask, ts*mask, ps*mask
    if compute_gradient:
        jx, jy = jx*mask, jy*mask
    iwe, iwe_drv = events_to_image_drv(xs, ys, ps, jx, jy,
            interpolation='bilinear', compute_gradient=compute_gradient)
    return iwe, iwe_drv


# --------------------------------------------------------------------------
# Objective functions
# --------------------------------------------------------------------------
class objective_function(ABC):

    def __init__(self, name="template", use_polarity=True,
            has_derivative=True, default_blur=1.0):
        self.name = name
        self.use_polarity = use_polarity
        self.has_derivative = has_derivative
        self.default_blur = default_blur
        super().__init__()

    @abstractmethod
    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Define the warp function. Either give the params and the events or give a
        precomputed iwe (if xs, ys, ts, ps are given, iwe is not necessary).
        """
        #if iwe is None:
        #    iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size,
        #            use_polarity=self.use_polarity, compute_gradient=False)
        #blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        #if blur_sigma > 0:
        #    iwe = gaussian_filter(iwe, blur_sigma)
        #loss = compute_loss_here...
        #return loss
        pass

    @abstractmethod
    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        If your warp function has it, define the gradient (otherwise set has_derivative to False
        and numeric grads will be used). Either give the params and the events or give a
        precomputed iwe and d_iwe (if xs, ys, ts, ps are given, iwe, d_iwe is not necessary).
        """
        #if iwe is None or d_iwe is None:
        #    iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size,
        #            use_polarity=self.use_polarity, compute_gradient=True)
        #blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        #if blur_sigma > 0:
        #    d_iwe = gaussian_filter(d_iwe, blur_sigma)

        #gradient = []
        #for grad_dim in range(d_iwe.shape[0]):
        #    gradient.append(compute_gradient_here...)
        #grad = np.array(gradient)
        #return grad
        pass
    
class variance_objective(objective_function):
    """
    Variance objective from 'Gallego, Accurate Angular Velocity Estimation with an Event Camera, RAL'17'
    """
    def __init__(self):
        self.use_polarity = True
        self.name = "variance"
        self.has_derivative = True
        self.default_blur=1.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by var(g(x)) where g(x) is IWE
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        loss = np.var(iwe-np.mean(iwe))
        return -loss

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        Gradient given by 2*(g(x)-mu(g(x))*(g'(x)-mu(g'(x))) where g(x) is the IWE
        """
        if iwe is None or d_iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=True)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            d_iwe = gaussian_filter(d_iwe, blur_sigma)

        gradient = []
        zero_mean = 2.0*(iwe-np.mean(iwe))
        for grad_dim in range(d_iwe.shape[0]):
            mean_jac = d_iwe[grad_dim]-np.mean(d_iwe[grad_dim])
            gradient.append(np.mean(zero_mean*(d_iwe[grad_dim]-np.mean(d_iwe[grad_dim]))))
        grad = np.array(gradient)
        return -grad

class r1_objective(objective_function):
    """
    R1 objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """
    def __init__(self, p=3):
        self.name = "r1"
        self.use_polarity = False
        self.has_derivative = False
        self.p = p
        self.default_blur = 1.0
        self.last_sosa = 0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by SOS and SOSA combined
        """
        if iwe is None:
            iwe, d_iwe = get_iwe(params, xs, ys, ts, ps, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        if blur_sigma > 0:
            iwe = gaussian_filter(iwe, blur_sigma)
        sos = np.mean(iwe*iwe)
        exp = np.exp(-self.p*iwe.astype(np.double))
        sosa = np.sum(exp)
        if sosa > self.last_sosa:
            return -sos
        self.last_sosa = sosa
        return -sos*sosa

    def evaluate_gradient(self, params=None, xs=None, ys=None, ts=None, ps=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, d_iwe=None):
        """
        No derivative known
        """
        return None
