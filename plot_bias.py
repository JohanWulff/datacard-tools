import ROOT
import numpy as np
import argparse

ROOT.gROOT.SetBatch(True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot pull distribution.")
parser.add_argument("r_truth", type=float, help="Truth value of r.")
parser.add_argument("root_file", type=str, help="Path to the .root file (output from multiDimFit with toys).")
parser.add_argument("r_min", type=float, help="Minimum value of r.")
parser.add_argument("r_max", type=float, help="Maximum value of r.")
args = parser.parse_args()

r_truth = args.r_truth
root_file = args.root_file
r_min = args.r_min
r_max = args.r_max

# Open file with fits
c = ROOT.TChain("limit")
c.Add(root_file)

hist_pull = ROOT.TH1F("pull", "Pull distribution: truth=%lf" % (r_truth), 100, -5, 5)
hist_pull.GetXaxis().SetTitle("Pull = (r_{truth}-r_{fit})/#sigma_{fit}")
hist_pull.GetYaxis().SetTitle("Entries")

N_toys = int(c.GetEntries() / 3)  # every toy has 3 r-values: r_hi = r_fit-err_low , r_fit , r_lo = r_fit+err_high

for i_toy in range(N_toys):
    # Best-fit value
    c.GetEntry(i_toy * 3)
    r_fit = getattr(c, "r")

    # -1 sigma value
    c.GetEntry(i_toy * 3 + 1)
    r_lo = getattr(c, "r")

    # +1 sigma value
    c.GetEntry(i_toy * 3 + 2)
    r_hi = getattr(c, "r")

    diff = r_truth - r_fit
    # Use uncertainty depending on where mu_truth is relative to mu_fit
    if diff > 0:
        sigma = abs(r_hi - r_fit)  # when r_fit < r_truth, it needs to go upward to approach r_fit, thus the up-error is used
    else:
        sigma = abs(r_lo - r_fit)  # when r_fit > r_truth, it needs to go downward to approach r_fit, thus the low-error is used

    if abs(r_hi - r_max) > 1e-3:  # Minos didn't return the rMax properly
        if abs(r_lo + r_max) > 1e-3:  # Minos didn't return the rMin properly
            if abs(sigma) > 1e-3:  # Errors returned by Minos too small, indicating again fit issues
                hist_pull.Fill(diff / sigma)
            else:
                print("r_fit: %f, sigma: %f is too small" % (r_fit, sigma))
        else:
            print("r: %f, r_lo: %f touches rMin" % (r_fit, r_lo))
    else:
        print("r: %f, r_hi: %f touches rMax" % (r_fit, r_hi))

canv = ROOT.TCanvas()
hist_pull.Draw()
print(hist_pull.GetEntries())

# Fit Gaussian to pull distribution
ROOT.gStyle.SetOptFit(111)
hist_pull.Fit("gaus")

canv.SaveAs("pull_r%d.png" % (r_truth))
