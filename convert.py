import torch
import torch.onnx
from models import create_model
import os
from options.test_options import TestOptions
from data import create_dataset

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # A model class instance (class not shown)
    model = create_model(opt)
    #model.setup(opt)
    model = model.netG
    #model.setup(opt)
    #model.test()
    #model.state_dict()

    # Load the weights from a file (.pth usually)
    state_dict = torch.load("./checkpoints/paths_pix2pix/96_net_G.pth")

    # Load the weights now into a model net architecture defined by our class
    model.module.load_state_dict(state_dict)

    # Create the right input shape (e.g. for an image)
    dummy_input = torch.randn(1, 3, 256, 256).cuda()

    #model = create_model(opt)      # create a model given opt.model and other options

    #model.setup(opt)
    #dummy_input = dataset[1]
    #print(dataset.data[1])
    torch.onnx.export(model.module, dummy_input, "onnx_model_name.onnx")

