from __future__ import annotations

from constants import _PI as PI
from mass_models import truncGausslowerupper_at_lpdf_safe

from pytensor_utils import logsumexp2


def _spins_as_list(spins, spin_model):
    """
    spins: array (N, nspin) or already a list/tuple
    returns: list of spin vectors (each shape (N,))
    """
    # already a list -> return as-is
    if isinstance(spins, (list, tuple)):
        return list(spins)

    # otherwise assume matrix
    if spin_model in ("default", "default_gauss"):
        return [spins[:, 0], spins[:, 1], spins[:, 2], spins[:, 3]]
    elif spin_model in ("chieffchip", "chieffchip_uc"):
        return [spins[:, 0], spins[:, 1]]
    elif spin_model == "none":
        return []
    else:
        raise NotImplementedError(f"spin_model={spin_model}")

# ---------------------------------------------------------------------
# Default gauss
# ---------------------------------------------------------------------


def logpdf_default_spin_gauss(bk, theta, lambdaBBHspin):
    """
    Backend-agnostic version of your legacy logpdf_default_spin_gauss.

    theta = (chi1, chi2, cost1, cost2)
    lambdaBBHspin = (muChi, sigmaChi, zeta, sigmat)
    """
    chi1, chi2, cost1, cost2 = theta
    muChi, sigmaChi, zeta, sigmat = lambdaBBHspin

    # amplitudes: truncated Gaussian on [0,1]
    lpdfs1 = truncGausslowerupper_at_lpdf_safe(bk, chi1, muChi, sigmaChi, xmin=0.0, xmax=1.0, truncate=False)
    lpdfs2 = truncGausslowerupper_at_lpdf_safe(bk, chi2, muChi, sigmaChi, xmin=0.0, xmax=1.0, truncate=False)
    logpdfampl = lpdfs1 + lpdfs2

    # cos tilts: "Gaussian around 1" piece
    # NOTE: matches your exact legacy expression
    norm = bk.log(sigmat) + bk.log(bk.erf(bk.sqrt(2.0) / sigmat))
    lpdfcos1_gauss = -0.5 * (1.0 - cost1) ** 2 / (sigmat ** 2) - norm
    lpdfcos2_gauss = -0.5 * (1.0 - cost2) ** 2 / (sigmat ** 2) - norm

    # mixture between aligned (Gaussian) and isotropic
    t1 = bk.log(2.0) + bk.log(zeta) - bk.log(PI) + lpdfcos1_gauss + lpdfcos2_gauss
    t2 = bk.log(1.0 - zeta) - bk.log(4.0)

    return logpdfampl + logsumexp2(bk, t1, t2)