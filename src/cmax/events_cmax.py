import numpy as np
import scipy.optimize as opt
from src.cmax import objectives
from src.cmax import warps

def optimize_contrast(xs, ys, ts, ps, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None,
        numeric_grads=False, blur_sigma=None, img_size=(180, 240), init=None):
    """
    Optimize contrast for a set of events
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp_function (function) The function with which to warp the events
    objective (objective class object) The objective to optimize
    optimizer (function) The optimizer to use
    x0 (np array) The initial guess for optimization
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor
    blur_sigma (float) Size of the blurring kernel. Blurring the images of warped events can
        have a large impact on the convergence of the optimization.

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    args = (xs, ys, ts, ps, warp_function, img_size, blur_sigma)
    x0 = np.array([0,0])
    if init is None:
        x0 = np.zeros(warp_function.dims)
    else:
        x0 = np.array(init)
    if numeric_grads:
        argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False)
    else:
        argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=False)
    return argmax

def optimize(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240), init=None, blur=None):
    """
    Optimize contrast for a set of events. Uses optimize_contrast() for the optimiziation, but allows
    blurring schedules for successive optimization iterations.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, init=init, blur_sigma=blur, img_size=img_size)
    return argmax_an
