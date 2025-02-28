class WGAN_GP_Loss:
    def __init__(self, lambda_gp=10.0):
        self.lambda_gp = lambda_gp

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.rand(real_samples.F.shape[0], 1).to(real_samples.device)
        interpolates = ME.SparseTensor(
            features=alpha * real_samples.F + (1 - alpha) * fake_samples.F,
            coordinate_map_key=real_samples.coordinate_map_key,
            coordinate_manager=real_samples.coordinate_manager
        )
        d_interpolates = D(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates.F,
            inputs=interpolates.F,
            grad_outputs=torch.ones_like(d_interpolates.F),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
