    
import torch
import torch.nn as nn
import torch.distributions as dist


class SimpleGaussianDecoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        # משתמשים ב-Backbone המשותף
        self.backbone = MNISTDecoderBackbone(latent_dim)
        
        # Head: מפיק תמונה (Mean)
        self.head_mu = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        # אנחנו רוצים שהשונות תהיה סקלר נלמד (או קבוע), אבל זהה לכל הפיקסלים
        # כדי לא לאפשר למודל "לרמות" ולהתעלם מפיקסלים קשים.
        self.raw_scale = nn.Parameter(torch.tensor(0.0)) # start at sigma=1

    def forward(self, z):
        features = self.backbone(z) # [B, 32, 28, 28]
        
        # 1. Mean
        # Sigmoid כדי להבטיח שהממוצע בטווח [0,1] כמו התמונות
        mu = torch.sigmoid(self.head_mu(features)) 
        
        # 2. Scale (Diagonal)
        # משדרים את הסקלר לכל התמונה [B, 1, 28, 28]
        scale = (nn.functional.softplus(self.raw_scale) + 1e-4).expand_as(mu)
        
        return mu, scale
    
    def get_distribution(self, z) -> dist.Independent:
        mu, scale = self(z)

        # תיקון קריטי: ndims=3 כדי לכלול את (C, H, W) כאירוע אחד
        return dist.Independent(
            dist.Normal(loc=mu, scale=scale), reinterpreted_batch_ndims=3
        )
    
class GaussianCopulaDecoder(nn.Module):
    def __init__(self, latent_dim=4, output_channels=1, rank=1):
        super().__init__()
        self.rank = rank
        
        # שימוש חוזר ב-Backbone (אותו מספר פרמטרים בדיוק עד ה-Head)
        self.backbone = MNISTDecoderBackbone(latent_dim)
        
        # Head 1: Mean (זהה ל-MF)
        self.head_mu = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        
        # Head 2: Factors W (התוספת לקורלציות)
        # מוציא 'rank' ערוצים, באותו גודל מרחבי (28x28)
        self.head_factors = nn.Conv2d(32, rank, kernel_size=3, padding=1)

        # Global Scalar Variance (Omega)
        self.log_omega = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        features = self.backbone(z) # [B, 32, 28, 28]
        
        # 1. Mean
        mu = torch.sigmoid(self.head_mu(features)) # [B, 1, 28, 28]
        
        # חייבים לשטח ל-784 בשביל LowRankMultivariateNormal
        mu_flat = mu.view(mu.shape[0], -1) 
        
        # 2. Diagonal Variance (Omega)
        omega = torch.nn.functional.softplus(self.log_omega) + 1e-4
        diag_scale = omega.sqrt() * torch.ones_like(mu_flat)

        # 3. Factors (W)
        factors = self.head_factors(features) # [B, Rank, 28, 28]
        
        # עיצוב מחדש ל-LowRankMultivariateNormal: [Batch, Pixels, Rank]
        # אנחנו צריכים שהמימד האחרון יהיה ה-Rank
        factors = factors.view(factors.shape[0], self.rank, -1) # [B, Rank, 784]
        factors = factors.permute(0, 2, 1) # [B, 784, Rank]

        return mu_flat, diag_scale, factors

    def get_distribution(self, z):
        mu_flat, diag_scale, factors = self.forward(z)
        cov_diag = diag_scale.pow(2)
        
        # התפלגות התומכת בקורלציות (דורשת וקטור שטוח)
        return dist.LowRankMultivariateNormal(
            loc=mu_flat,
            cov_factor=factors,
            cov_diag=cov_diag
        )


class MNISTDecoderBackbone(nn.Module):
    """
    Input: Latent Vector [B, z_dim]
    Output: Spatial Feature Map [B, 32, 28, 28]
    """
    def __init__(self, latent_dim=4):
        super().__init__()
        self.reshape_dim = (64, 7, 7)
        self.flatten_dim = 64 * 7 * 7
        
        # Projection
        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        
        # Spatial Reconstruction
        self.net = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            # nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            # nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2)
            # הפלט הוא Feature Map עם 32 ערוצים, לא התמונה הסופית!
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, *self.reshape_dim)
        return self.net(h)
