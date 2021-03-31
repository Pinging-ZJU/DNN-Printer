import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def output_shape(array):
    out = 1
    for i in array:
        out = out * i
    return out

def iterlen(x):
    return sum(1 for _ in x)

def DNN_printer(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(DNN_printer)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            DNN_printer[m_key] = OrderedDict()
            DNN_printer[m_key]["input_shape"] = list(input[0].size())
            DNN_printer[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                DNN_printer[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                DNN_printer[m_key]["output_shape"] = list(output.size())
                DNN_printer[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                DNN_printer[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            DNN_printer[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
  
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # create properties
    DNN_printer = OrderedDict()
    hooks = []

    # register hook
    # model.apply(register_hook)

    def apply_hook(module):
        for name, submodule in module.named_children():
            if (iterlen(submodule.named_children()) == 0):
                submodule.apply(register_hook)
            else:
                apply_hook(submodule)

    apply_hook(model)

    model(*x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print("------------------------------Happy every day! :)---------------------------------")
    print("-----------------------------Author: Peiyi & Ping---------------------------------")
    line_new = "{:>20} {:>20} {:>15} {:>13} {:>15}".format("Layer (type)", "Output Shape", "O-Size(MB)" ,"Param #","P-Size(MB)")
    print(line_new)
    print("==================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in DNN_printer:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>20} {:>10} {:>13} {:>15}".format(
            layer,
            str(DNN_printer[layer]["output_shape"]),
            str((output_shape(DNN_printer[layer]["output_shape"])) * 4 / 1024 / 1024 )+" MB",
            "{0:,}".format(DNN_printer[layer]["nb_params"]),
            str(float(DNN_printer[layer]["nb_params"]) * 4 / 1024 / 1024) + " MB",
        )
        total_params += DNN_printer[layer]["nb_params"]
        total_output += np.prod(DNN_printer[layer]["output_shape"])
        if "trainable" in DNN_printer[layer]:
            if DNN_printer[layer]["trainable"] == True:
                trainable_params += DNN_printer[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
