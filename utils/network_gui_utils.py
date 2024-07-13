import torch
from gaussian_renderer import render, network_gui

class NetworkGUI:
    @staticmethod
    def init(ip, port):
        network_gui.init(ip, port)
    
    @staticmethod
    def render(
        dataset,
        opt,
        pipe,
        gaussians,
        background,
        iteration,
    ):
        if network_gui.conn is None:
            network_gui.try_connect()

        while network_gui.conn is not None:
            try:
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()

                if custom_cam is not None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]

                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                else:
                    net_image_bytes = None

                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None
