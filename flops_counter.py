import torch
import torch.nn as nn
import numpy as np


class DConv2d(nn.Module):

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=1, bias=None):
        super(DConv2d, self).__init__()
        self.stride = stride

    def __repr__(self):
        s = '{}(weight_size={}, '.format(self.__class__.__name__, self.weight.shape)
        bias_shape = None if self.bias is None else self.bias.shape
        if hasattr(self, 'projection'):
            s += 'projection_size={}, bias_size={}, '.format(self.projection.shape,bias_shape)
        else:
            s += 'bias_size={}, '.format(bias_shape)
        if hasattr(self, 'projection2'):
            s += 'projection2_size={}, '.format(self.projection2.shape)
        return s


    def set_params(self, input):
        self.weight = input['weight']
        self.bias = input['bias']
        if 'projection' in input:
            self.projection = input['projection']
        if 'projection2' in input:
            self.projection2 = input['projection2']
        self.padding = self.weight.shape[-1] // 2

    def forward(self, x):
     
        if hasattr(self, 'projection'):
            bias_shape = None if self.bias is None else self.bias.shape[0]

            m = F.conv2d(x, weight=self.weight, padding=self.padding, stride=self.stride)
            y = F.conv2d(m, weight=self.projection, bias=self.bias)

            if hasattr(self, '__store_input__'):
                self.feature_map_storage(x, m, y)
                
        else:
            y = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding, stride=self.stride)

        if hasattr(self, 'projection2'):
            y = F.conv2d(x, weight=self.projection2)


        return y

    def feature_map_storage(self, x, m, y):
        features = {}
        if self.__store_input__:
            features['input'] = x
        if self.__store_middle__:
            features['middle'] = m.norm(dim=(2, 3)).mean(dim=0, keepdim=True)
        if self.__store_output__:
            features['output'] = y
        torch.save(features, os.path.join(self.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
        print('{} {}, Data Batch {}, Device {}'.format(self.__class__.__name__, self.__count_layer__, count_data, torch.cuda.current_device()))

    def feature_map_inter_norm(self, m):
        feat = m.detach().cpu().norm(dim=(2, 3)).mean(dim=0, keepdim=True)
        if hasattr(self, '__feature_map_norm__'):
            self.__feature_map_norm__ = torch.cat((self.__feature_map_norm__, feat), dim=0)
        else:
            self.__feature_map_norm__ = feat
        print(self.__class__.__name__, self.__store_input__, self.__store_output__, self.__store_middle__, id(self))





def get_model_flops(model, input_res, print_per_layer_stat=True, input_constructor=None):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        device = list(flops_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    flops_model.stop_flops_count()

    return flops_count


def get_model_parameters(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def get_model_complexity_info(model, input_res, print_per_layer_stat=True, as_strings=False, input_constructor=None):
    flops_count = get_model_flops(model, input_res, print_per_layer_stat, input_constructor)
    params_count = get_model_parameters(model)

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num):
    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + ' M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + ' k'
    else:
        return str(params_num)


def print_model_with_flops(model, units='GMac', precision=3):
    total_flops = model.compute_average_flops_cost()

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([flops_to_string(accumulated_flops_cost, units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model)
    model.apply(del_extra_repr)


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    # embed()
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):
        if isinstance(module, nn.Conv2d):
            module.__mask__ = mask
    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions
def is_supported_instance(module):

    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
                           nn.Linear,
                           nn.BatchNorm2d, nn.BatchNorm3d,
                           nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6,
                           DConv2d)):
        return True



    return False


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    if len(input.shape) == 1:
        batch_size = 1
        module.__flops__ += int(batch_size * input.shape[0] * output.shape[0])
    else:
        batch_size = input.shape[0]
        module.__flops__ += int(batch_size * input.shape[1] * output.shape[1])


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    module.affine
    input = input[0]
    # TODO: need to check here
    # batch_flops = np.prod(input.shape)
    # if module.affine:
    #     batch_flops *= 2
    # module.__flops__ += int(batch_flops)
    batch = input.shape[0]
    output_dims = input.shape[2:]
    channels = module.num_features
    batch_flops = batch * channels * np.prod(output_dims)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def deconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    input_height, input_width = input.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

    active_elements_count = batch_size * input_height * input_width
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        output_height, output_width = output.shape[2:]
        bias_flops = out_channels * batch_size * output_height * output_height
    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def dconv_flops_counter_hook(dconv_module, input, output):
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    m_channels, in_channels, kernel_dim1, _, = dconv_module.weight.shape
    out_channels, _, kernel_dim2, _, = dconv_module.projection.shape
    # groups = dconv_module.groups

    # filters_per_channel = out_channels // groups
    conv_per_position_flops1 = kernel_dim1 ** 2 * in_channels * m_channels
    conv_per_position_flops2 = kernel_dim2 ** 2 * out_channels * m_channels
    active_elements_count = batch_size * np.prod(output_dims)

    overall_conv_flops = (conv_per_position_flops1 + conv_per_position_flops2) * active_elements_count
    overall_flops = overall_conv_flops

    dconv_module.__flops__ += int(overall_flops)
    # dconv_module.__output_dims__ = output_dims


def conv_flops_counter_hook(conv_module, input, output):
    # embed()
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = np.prod(kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * np.prod(output_dims)

    overall_conv_flops = conv_per_position_flops * active_elements_count

    overall_flops = overall_conv_flops

    conv_module.__flops__ += int(overall_flops)
    conv_module.__output_dims__ = output_dims


# def conv_flops_counter_hook(conv_module, input, output):
#     # Can have multiple inputs, getting the first one
#     input = input[0]
#
#     batch_size = input.shape[0]
#     output_dims = list(output.shape[2:])
#
#     kernel_dims = list(conv_module.kernel_size)
#     in_channels = conv_module.in_channels
#     out_channels = conv_module.out_channels
#     groups = conv_module.groups
#
#     filters_per_channel = out_channels // groups
#     conv_per_position_flops = np.prod(kernel_dims) * in_channels * filters_per_channel
#
#     active_elements_count = batch_size * np.prod(output_dims)
#
#     if conv_module.__mask__ is not None:
#         # (b, 1, h, w)
#         flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
#         active_elements_count = flops_mask.sum()
#
#     overall_conv_flops = conv_per_position_flops * active_elements_count
#
#     bias_flops = 0
#
#     if conv_module.bias is not None:
#
#         bias_flops = out_channels * active_elements_count
#
#     overall_flops = overall_conv_flops + bias_flops
#
#     conv_module.__flops__ += int(overall_flops)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module, assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, nn.ConvTranspose2d):
            handle = module.register_forward_hook(deconv_flops_counter_hook)
        elif isinstance(module, nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d,
                                 nn.AdaptiveAvgPool2d, nn.MaxPool3d, nn.AvgPool3d,
                                 nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d)):
            handle = module.register_forward_hook(pool_flops_counter_hook)
        elif isinstance(module, nn.Upsample):
            handle = module.register_forward_hook(upsample_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if is_supported_instance(module):
        module.__mask__ = None
