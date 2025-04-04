import torch
import numpy as np
import math

EPS = 1e-6
class MixedNormalCRPS(torch.nn.Module):
    def __init__(self, reduce: bool = True, c: float = np.log(0.01)):
        super(MixedNormalCRPS, self).__init__()
        self.reduce = reduce
        self.c = c

    def crps(
        self,
        prediction: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

        :param torch.Tensor y: observed data
        :param torch.Tensor prediction: Tensor of prediction, containing parameters [mu, sigma, p]

        :return tensor: CRPS value
        :rtype torch.Tensor
        """
        mask = ~torch.isnan(y)

        mu, sigma, p = torch.split(prediction, 1, dim=1)
        y = y.unsqueeze(1)

        if isinstance(p, float):
            p = torch.tensor([p])

        if isinstance(self.c, float):
            c = torch.tensor([self.c]).to(y.device)

        # Mask NaNs
        mu = mu[mask]
        sigma = sigma[mask]
        y = y[mask]
        p = p[mask]

        # Transformed variables
        y_transformed = (y - mu) / sigma
        c_transformed = (c - mu) / sigma

        # Define normal distribution
        normal = torch.distributions.Normal(loc=0, scale=1)

        # Compute Terms
        P_c = p + (1 - p) * normal.cdf(c_transformed)

        # Compute CRPS
        t1 = y_transformed * (2 * (p + (1 - p) * normal.cdf(y_transformed)) - 1)
        t2 = -c_transformed * torch.pow(P_c, 2)
        t3 = 2 * (1 - p) * (-normal.log_prob(c_transformed).exp()) * P_c
        t4 = -2 * (1 - p) * (-normal.log_prob(y_transformed).exp())
        t5 = (
            2
            * torch.pow(1 - p, 2)
            * (-1 / (2 * math.sqrt(math.pi)))
            * (1 - normal.cdf(math.sqrt(2) * c_transformed))
        )

        crps = sigma * (t1 + t2 + t3 + t4 + t5)

        if self.reduce:
            crps = torch.mean(crps)
        return crps


class MixedLoss(torch.nn.Module):
    def __init__(self, grad_u: bool, xi:float, u = None, reduce: bool = True,  t:float =5, c=np.log(0.01)):
        super(MixedLoss, self).__init__()
        self.reduce = reduce
        self.c = c
        self.grad_u = grad_u
        self.u = u
        self.xi = xi
        self.t = t

    def _gpd(self, x_re: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Calculates the GPD distribution function for xi unequal 0

        :param torch.Tensor x: Tensor of values normalized with u and sigma
        :param torch.Tensor xi: Tensor of shape parameter

        :return tensor: GPD value
        :rtype torch.Tensor
        """
        cdf = torch.where(x_re <= 0, 0, 1 - (1 + xi * x_re).pow(-1 / xi))
        return cdf

    def _delta_u(
        self,
        u_transformed: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the delta_u parameter for the mixed distribution

        :param torch.Tensor u: Tensor of threshold
        :param torch.Tensor p: Tensor of discrete part of the normal distribution in [0,1]

        :return tensor: delta_u value
        :rtype torch.Tensor
        """
        dist = torch.distributions.Normal(loc=0, scale=1)
        delta_u = p + (1 - p) * dist.cdf(u_transformed)

        return delta_u

    def _pareto_crps(
        self,
        y: torch.Tensor,
        u: torch.Tensor,
        m: torch.Tensor,
        sigma: torch.Tensor,
        xi: torch.Tensor,
    ) -> torch.Tensor:
        y_transformed = (y - u) / sigma
        cdf = self._gpd(y_transformed, xi)
        crps = sigma * (
            torch.abs(y_transformed)
            - 2 * (1 - m) / (1 - xi) * (1 - torch.pow(1 - cdf, 1 - xi))
            + torch.pow(1 - m, 2) / (2 - xi)
        )
        return crps

    def _mixed_normal_crps(
        self,
        y_transformed: torch.Tensor,
        p: torch.Tensor,
        c_transformed: float,
        u_transformed,
        sigma: torch.Tensor,
    ):

        # Define normal distribution
        normal = torch.distributions.Normal(loc=0, scale=1)

        # Compute Terms
        P_c = p + (1 - p) * normal.cdf(c_transformed)
        P_u = (1 - p) * (1 - normal.cdf(u_transformed))

        # Compute CRPS
        t1 = y_transformed * (2 * (p + (1 - p) * normal.cdf(y_transformed)) - 1)
        t2 = -c_transformed * torch.pow(P_c, 2) + u_transformed * torch.pow(P_u, 2)
        t3 = (
            2 * (1 - p) * (-normal.log_prob(c_transformed).exp()) * P_c
            + 2 * (1 - p) * (-normal.log_prob(u_transformed).exp()) * P_u
        )
        t4 = -2 * (1 - p) * (-normal.log_prob(y_transformed).exp())
        t5 = (
            2
            * torch.pow(1 - p, 2)
            * (-1 / (2 * math.sqrt(math.pi)))
            * (
                normal.cdf(math.sqrt(2) * u_transformed)
                - normal.cdf(math.sqrt(2) * c_transformed)
            )
        )

        crps = sigma * (t1 + t2 + t3 + t4 + t5)
        return crps

    def _mixed_normal_crps_upper(
        self,
        p: torch.Tensor,
        c_transformed: float,
        u_transformed,
        sigma: torch.Tensor,
    ):

        # Define normal distribution
        normal = torch.distributions.Normal(loc=0, scale=1)

        # Compute Terms
        P_c = p + (1 - p) * normal.cdf(c_transformed)
        P_u = (1 - p) * (1 - normal.cdf(u_transformed))

        # Compute CRPS
        t1 = u_transformed
        t2 = -c_transformed * torch.pow(P_c, 2) + u_transformed * torch.pow(P_u, 2)
        t3 = (
            2 * (1 - p) * (-normal.log_prob(c_transformed).exp()) * P_c
            + 2 * (1 - p) * (-normal.log_prob(u_transformed).exp()) * P_u
        )
        t4 = -2 * (
            (1 - p) * (-normal.log_prob(u_transformed).exp()) + u_transformed * P_u
        )
        t5 = (
            2
            * torch.pow(1 - p, 2)
            * (-1 / (2 * math.sqrt(math.pi)))
            * (
                normal.cdf(math.sqrt(2) * u_transformed)
                - normal.cdf(math.sqrt(2) * c_transformed)
            )
        )

        crps = sigma * (t1 + t2 + t3 + t4 + t5)
        return crps

    def crps(
        self,
        prediction: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

        :param torch.Tensor y: observed data
        :param torch.Tensor prediction: Tensor of prediction, containing parameters [mu, sigma, p]

        :return tensor: CRPS value
        :rtype torch.Tensor
        """
        mask = ~torch.isnan(y)

        if self.grad_u == True:
            mu, sigma, p, sigma_u, u = torch.split(prediction, 1, dim=1)
            u = u[mask]
        else:
            mu, sigma, p, sigma_u = torch.split(prediction, 1, dim=1)
            u = torch.tensor([self.u]).to(y.device)

        y = y.unsqueeze(1)

        if isinstance(p, float):
            p = torch.tensor([p])

        if isinstance(self.c, float):
            c = torch.tensor([self.c]).to(y.device)

        if isinstance(self.xi, float):
            xi = torch.tensor([self.xi]).to(y.device)

        # Mask NaNs
        mu = mu[mask]
        sigma = sigma[mask]
        y = y[mask]
        p = p[mask]
        sigma_u = sigma_u[mask]

        # Transformed variables
        c_transformed = (c - mu) / sigma
        u_transformed = (u - mu) / sigma
        y_transformed = (y - mu) / sigma

        # Calculate threshold
        m_u = self._delta_u(u_transformed=u_transformed, p=p)
        loss_1 = self._mixed_normal_crps(
            y_transformed=y_transformed,
            p=p,
            c_transformed=c_transformed,
            u_transformed=u_transformed,
            sigma=sigma,
        ) + self._pareto_crps(
            y=u, u=u, m=m_u, sigma=sigma_u, xi=xi
        )
        loss_2 = self._pareto_crps(
            y=y, u=u, m=m_u, sigma=sigma_u, xi=xi
        ) + self._mixed_normal_crps_upper(
            p=p, c_transformed=c_transformed, u_transformed=u_transformed, sigma=sigma
        )

        if self.grad_u:
            crps = torch.sigmoid((u - y) * self.t) * (loss_1 - loss_2) + loss_2
        else:
            crps = torch.where(y < u, loss_1, loss_2)

        if self.reduce:
            crps = torch.mean(crps)
        return crps


def crps_no_avg(mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    mu, sigma = torch.split(mu_sigma, 1, dim=-1)
    y = y.view((-1, 1))  # make sure y has the right shape
    pi = np.pi  
    omega = (y - mu) / sigma
    # PDF of normal distribution at omega
    pdf = 1 / (torch.sqrt(torch.tensor(2 * pi))) * torch.exp(-0.5 * omega**2)

    # Source:
    # https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    cdf = 0.5 * (1 + torch.erf(omega / torch.sqrt(torch.tensor(2))))

    crps_score = sigma * (
        omega * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(pi))
    )
    return crps_score


def crps_active_stations(
    mu_sigma: torch.Tensor, y: torch.Tensor, active_stations: torch.Tensor
) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) for all stations which have valid measurements

    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df
    :param torch.Tensor active_stations: tensor of active stations

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    active_stations = active_stations.to(torch.bool)
    active_stations = ~active_stations

    mu_sigma = mu_sigma[active_stations]
    y = y[active_stations]
    crps_score = crps_averaged(mu_sigma=mu_sigma, y=y)
    return crps_score


def crps_averaged(mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    crps_score = crps_no_avg(mu_sigma=mu_sigma, y=y)
    return torch.mean(crps_score)


class NormalCRPS(torch.nn.Module):
    """Source: Höhlein et. al (2024) Postprocessing of Ensemble Weather Forecasts Using
    Permutation-Invariant Neural Networks
    https://github.com/khoehlein/Permutation-invariant-Postprocessing/blob/main/model/loss/losses.py
    """

    def __init__(self):
        super(NormalCRPS, self).__init__()
        self._inv_sqrt_pi = 1 / torch.sqrt(torch.tensor(np.pi))
        self.dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    def crps(self, prediction: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

        :param torch.Tensor mu_sigma: tensor of mean and standard deviation
        :param torch.Tensor y: observed df

        :return tensor: CRPS value
        :rtype torch.Tensor
        """
        mask = ~torch.isnan(y)
        mu, sigma = torch.split(prediction, 1, dim=1)
        y = y.unsqueeze(1)

        mu = mu[mask]
        sigma = sigma[mask]
        y = y[mask]

        z_red = (y - mu) / sigma

        cdf = self.dist.cdf(z_red)
        pdf = torch.exp(self.dist.log_prob(z_red))
        crps = sigma * (z_red * (2.0 * cdf - 1.0) + 2.0 * pdf - self._inv_sqrt_pi)
        crps_score = torch.mean(crps)
        return crps_score


# Test
if __name__ == "__main__":
    y = torch.rand((3, 5, 1))
    mu_normal = torch.zeros((3, 5, 1))
    sigma_normal = torch.ones((3, 5, 1))
    delta_0 = torch.zeros((3, 5, 1)) + 0.5

    # Test
    crps_2 = MixedNormalCRPS(reduce=True)
    crps_value_2 = crps_2.crps(y, mu_normal, sigma_normal, delta_0)
    print(crps_value_2)
