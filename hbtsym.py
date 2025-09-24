# coding: utf-8

from dataclasses import dataclass

from law.util import multi_match


symmetrize_systematics = {
    "CMS_scale_t_*",
    "CMS_res_j_*",
    "CMS_scale_j_*",
    "CMS_res_e",
    "CMS_scale_e",
    "CMS_btag_*",
    # "CMS_bbtt_eff_xbb_pnet_*",
}


@dataclass
class BinContent:
    # bin contents
    b: int
    n: float
    # relative statistical bin errors
    err_n: float
    err_u: float
    err_d: float
    # up/down differences to nominal
    diff_u: float
    diff_d: float
    # state information
    symmetrize: bool = False

    @property
    def u(self) -> float:
        return self.n + self.diff_u

    @property
    def d(self) -> float:
        return self.n + self.diff_d

    @property
    def rel_diff_u(self) -> float:
        return (self.diff_u / self.n) if self.n != 0 else 0.0

    @property
    def rel_diff_d(self) -> float:
        return (self.diff_d / self.n) if self.n != 0 else 0.0


def symmetrize(process_name, syst_name, nom_hist, down_hist, up_hist):
    # settings
    epsilon = 1e-5
    n_bins_to_check = 1000  # how many bins to check, counted from the right most bin
    trustworty_stat_unc = 0.05  # relative statistical uncertainty below which to trust shifts
    zero_rel_unc = 0.001  # threshold below which relative differences are considered zero
    large_rel_unc = 0.05  # threshold above which relative differences are considered large
    asym_ratio = 5  # ratio between two relative shifts above which to consider them asymmetric

    # helper to get bin info
    def read_bin(b: int) -> BinContent:
        u = up_hist.GetBinContent(b)
        d = down_hist.GetBinContent(b)
        return BinContent(
            b=b,
            n=(n := nom_hist.GetBinContent(b)),
            # relative bin errors
            err_n=abs(nom_hist.GetBinError(b) / n) if n != 0 else 0.0,
            err_u=abs(up_hist.GetBinError(b) / u) if u != 0 else 0.0,
            err_d=abs(down_hist.GetBinError(b) / d) if d != 0 else 0.0,
            # compute differences
            diff_u=u - n,
            diff_d=d - n,
        )

    # read all bins and prepare values
    n_bins = nom_hist.GetNbinsX()
    bin_contents = {}
    for b in range(1, n_bins + 1):
        c = bin_contents[b] = read_bin(b)
        # keep all values as they are if up/down shifts have sufficiently small statistical uncertainties
        if c.err_u < trustworty_stat_unc and c.err_d < trustworty_stat_unc:
            continue
        # treat extremely tiny differences as zero
        if abs(c.rel_diff_u) < zero_rel_unc:
            c.diff_u = 0.0
        if abs(c.rel_diff_d) < zero_rel_unc:
            c.diff_d = 0.0
        # if they are large and highly asymmetric, consider the smaller one zero
        if abs(c.rel_diff_u) > large_rel_unc and c.rel_diff_d != 0 and abs(c.rel_diff_u / c.rel_diff_d) > asym_ratio:
            c.diff_d = 0.0
        if abs(c.rel_diff_d) > large_rel_unc and c.rel_diff_u != 0 and abs(c.rel_diff_d / c.rel_diff_u) > asym_ratio:
            c.diff_u = 0.0
        # decide whether or not the bin needs symmetrization: if the differences are on the same side _or_
        # one of them is zero (but not both!)
        c.symmetrize = (
            (c.diff_u > 0 and c.diff_d > 0) or
            (c.diff_u < 0 and c.diff_d < 0) or
            (c.diff_u == 0 and c.diff_d != 0) or
            (c.diff_u != 0 and c.diff_d == 0)
        )

    # iterate through bins starting from the right and adjust up/down differences if needed for symmetrization
    for b in range(n_bins, 0, -1):
        # only consider the right most n bins
        if (n_bins - b) >= n_bins_to_check:
            break
        # skip when not marked for symmetrization
        c = bin_contents[b]
        if not c.symmetrize:
            continue
        # actual symmetrization:
        # 1. when one diff is zero, re-use difference of the other one
        # 2. when both diffs are on the same side, flip one of them, maybe depending on continuity w.r.t. other bins
        if c.diff_u == 0:
            # case 1: up is zero
            c.diff_u = -c.diff_d
        elif c.diff_d == 0:
            # case 1: down is zero
            c.diff_d = -c.diff_u
        else:
            # case 2: one-sided
            # the effect that is moved to the other side is taken as the average
            flipped_diff = -(c.diff_u + c.diff_d) / 2
            # whether to flip the up or down shift depends on the neighboring, not-to-be-symmetrized bins
            neighbors = []
            if b > 1 and not bin_contents[b - 1].symmetrize:
                neighbors.append(bin_contents[b - 1])
            if b < n_bins and not bin_contents[b + 1].symmetrize:
                neighbors.append(bin_contents[b + 1])
            # get decision from neighbors (if they agree in the case of two of them):
            # 1 if their up variation is positive, -1 if negative, and 0 if undecided
            neighbor_sign = 0
            if len(neighbors) == 2:
                if neighbors[0].diff_u > 0 and neighbors[1].diff_u > 0:
                    neighbor_sign = 1
                elif neighbors[0].diff_u < 0 and neighbors[1].diff_u < 0:
                    neighbor_sign = -1
            elif len(neighbors) == 1:
                neighbor_sign = 1 if neighbors[0].diff_u > 0 else -1
            # actual flip
            if neighbor_sign == 1:
                # up variation is positive
                flip_attr = "diff_u" if c.diff_u < 0 else "diff_d"
            elif neighbor_sign == -1:
                # up variation is negative
                flip_attr = "diff_u" if c.diff_u > 0 else "diff_d"
            else:
                # flip the smaller one
                flip_attr = "diff_d" if abs(c.diff_u) > abs(c.diff_d) else "diff_u"
            setattr(c, flip_attr, flipped_diff)
        # finally adjust bins, no need to adapt errors as well as combine is not using them
        up_hist.SetBinContent(b, max(c.u, epsilon))
        down_hist.SetBinContent(b, max(c.d, epsilon))
        # mark the bin as not to be symmerized anymore
        c.symmetrize = False


def func(bin_name, process_name, nom_hist, systematic_shapes):
    if process_name == "data_obs":
        return

    for syst_name, (down_hist, up_hist) in systematic_shapes.items():
        if multi_match(syst_name, symmetrize_systematics):
            symmetrize(process_name, syst_name, nom_hist, down_hist, up_hist)
