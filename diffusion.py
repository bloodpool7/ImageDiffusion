import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import copy 
import model
from tqdm.auto import tqdm

class EMA: 
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = 'cpu', warmup: int = 0):
        self.model = model
        self.decay = decay 
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device) 
        self.num_updates = 0  
        self.warmup = warmup
        self.device = device

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update the ema model with the provided model weights.
        """
        for param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            if (self.num_updates < self.warmup):
                ema_param.data.copy_(param.data)
            else:
                ema_param.mul_(self.decay).add_(param, alpha=1 - self.decay)

        self.num_updates += 1 
        

class Image_Diffusion(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'): 
        super().__init__()
        self.T = num_timesteps 
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1 - self.betas 
        self.one_over_alphas = 1 / self.alphas
        self.one_over_sqrt_alphas = 1 / torch.sqrt(self.alphas)
        self.one_minus_alphas = 1 - self.alphas

        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.to(device)
    
    def forward_diffusion(self, x_0: torch.Tensor, timesteps: torch.Tensor, device: str = 'cpu'):
        """
        Returns the noised samples (x_t), added noise (epsilon), and the sinusoidal time embeddings for each timestep

        Expecting x_0 as [B, C, H, W] and timesteps as [B]

        Note: timesteps are 1-indexed, so we subtract 1 from the timesteps to get the correct alpha_bar
        """

        epsilon = torch.randn(x_0.shape, device=x_0.device)

        samples = self.sqrt_alpha_bars[timesteps - 1][:, None, None, None] * x_0 + self.sqrt_one_minus_alpha_bars[timesteps - 1][:, None, None, None] * epsilon 

        return samples.to(device), epsilon
    
    def train_one_epoch(self,
                        model: nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        ema: EMA,
                        conditioner: model.Conditioner,
                        unconditional: bool = False,
                        device: str = 'cpu',
                        epoch: int = 0):
        
        model.to(device) 
        model.train() 

        use_cuda_amp = isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None

        pbar = tqdm(data_loader, desc=f"Train Epoch {epoch}", unit="batch", dynamic_ncols=True)
        running_loss = 0.0

        for x_0, y in pbar:
            optimizer.zero_grad() 

            x_0 = x_0.to(device) 
            y = y.to(device)
            timesteps = torch.randint(1, self.T + 1, (x_0.shape[0],), device=device)
            cond = conditioner.get_training_condition(timesteps, y if not unconditional else None)

            x_t, epsilon = self.forward_diffusion(x_0, timesteps, device=device)
            if use_cuda_amp:
                with torch.autocast(device_type='cuda'):
                    epsilon_hat = model(x_t, cond)
                    loss = F.mse_loss(epsilon_hat, epsilon) 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.autocast(device_type=device):
                    epsilon_hat = model(x_t, cond)
                    loss = F.mse_loss(epsilon_hat, epsilon) 
                
                loss.backward()
                optimizer.step()
            ema.update(model)

            running_loss += loss.detach().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        mean_loss = running_loss / len(data_loader)
        pbar.set_postfix({"Epoch loss": f"{mean_loss:.4f}"})

        return mean_loss
    
    @torch.no_grad()
    def ddpm_reverse_diffusion(self, 
                        x_t: torch.Tensor,
                        epsilon_hat: torch.Tensor,
                        timesteps: torch.Tensor,
                        device: str = 'cpu'):
        """
        Returns the predicted samples (x_t_minus_one)

        Expecting x_t as [B, C, H, W] and timesteps as [B]

        Note: timesteps are 1-indexed, so we subtract 1 from the timesteps to get the correct alpha_bar
        """
        sqrt_betas_unfiltered = self.sqrt_betas[timesteps - 1]
        sqrt_betas = torch.where(timesteps == 1, torch.tensor(0, dtype=sqrt_betas_unfiltered.dtype), sqrt_betas_unfiltered)[:, None, None, None].to(device)
        sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[timesteps - 1][:, None, None, None].to(device)
        one_minus_alphas = self.one_minus_alphas[timesteps - 1][:, None, None, None].to(device)
        one_over_sqrt_alphas = self.one_over_sqrt_alphas[timesteps - 1][:, None, None, None].to(device)

        z = sqrt_betas * torch.randn_like(x_t)

        x_t_minus_one = one_over_sqrt_alphas * (x_t - (one_minus_alphas / sqrt_one_minus_alpha_bars) * epsilon_hat) + z

        return x_t_minus_one.to(device)

    @torch.no_grad()
    def sample(self, 
               model: nn.Module,
               conditioner: model.Conditioner,
               labels: torch.Tensor = None,
               guidance_scale: float = 7.0,
               num_samples: int = 1,
               device: str = 'cpu',
               image_shape=None):
        
        assert labels is None or labels.shape == (num_samples,), "Labels must be None or of length num_samples"

        model.to(device).eval()
        conditioner.to(device).eval()
        if image_shape is None:
            image_shape = (3, 32, 32)
        c, h, w = image_shape

        #initialize with pure noise
        x_t = torch.randn(num_samples, c, h, w, device=device)

        for t in tqdm(range(self.T, 0, -1), desc="Sampling", unit="timestep", dynamic_ncols=True):
            ts = torch.full((num_samples,), t, device=device, dtype=torch.int64)
            

            if (labels is None) or guidance_scale == 0.0:
                #Unconditional
                cond_u = conditioner.get_condition_vector(ts)
                eps_hat = model(x_t, cond_u)

            elif guidance_scale == 1.0:
                #Conditional without any guidance 
                cond_c = conditioner.get_condition_vector(ts, labels)
                eps_hat = model(x_t, cond_c)
            
            else:
                #Conditional with guidance 
                cond_u = conditioner.get_condition_vector(ts)
                cond_c = conditioner.get_condition_vector(ts, labels)
                
                x_in = torch.cat([x_t, x_t], dim=0)
                cond_in = torch.cat([cond_u, cond_c], dim=0)
                eps_u, eps_c = model(x_in, cond_in).chunk(2, dim=0)
                eps_hat = eps_u + guidance_scale * (eps_c - eps_u)

            #reverse diffusion (can be any sampling method)
            x_t = self.ddpm_reverse_diffusion(x_t, eps_hat, ts, device=device) # need a tensor t of shape [num_samples]

        return x_t

    #TODO: Need to update the evaluate method to factor in the conditioner
    @torch.no_grad()
    def evaluate(self,
                 model: nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: str = 'cpu'):
        
        model.to(device) 
        model.eval() 
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Eval", unit="batch", leave=False, dynamic_ncols=True)
            for x_0, _ in pbar:
                x_0 = x_0.to(device)
                timesteps = torch.randint(1, self.T + 1, (x_0.shape[0],), device=device)

                x_t, epsilon = self.forward_diffusion(x_0, timesteps, device=device)
                with torch.autocast(device_type=device):
                    epsilon_hat = model(x_t, timesteps)
                    loss = F.mse_loss(epsilon_hat, epsilon)
                
                running_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running_loss / num_batches if num_batches > 0 else 0.0
        