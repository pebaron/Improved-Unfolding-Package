#!/usr/bin/env python

import argparse
from matplotlib import pyplot as plt
import fbu
from fbu import Regularization
import ROOT
import math
from array import array
from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes
from ROOT import RooUnfoldSvd
from ROOT import RooUnfoldTUnfold
from ROOT import RooUnfoldIds
from ROOT import gRandom, TFile, TH1, TH1D, TH2, TH2D, cout
from ROOT import TGraph, TPad, gPad, TCanvas, TLegend, gStyle, gApplication, gStyle, kRed

parser = argparse.ArgumentParser(description="Unfolding code using RooUnfold and FBU package")
parser.add_argument('--rfile_data','-rfile_data', type=str, default="input.root")
parser.add_argument('--rfile_particle','-rfile_particle', type=str, default="input.root")
parser.add_argument('--rfile_matrix','-rfile_matrix', type=str, default="input.root")
parser.add_argument('--rfile_background','-rfile_background', type=str, default="input.root")
parser.add_argument('--h_data','-h_data', type=str, default="h_data")
parser.add_argument('--h_particle','-h_particle', type=str, default="h_particle")
parser.add_argument('--h_matrix','-h_matrix', type=str, default="h_matrix")
parser.add_argument('--h_background','-h_background', type=str, default="h_background")

parser.add_argument('--par','-p', type=float, default=1.25)
parser.add_argument('--title', '-title', type=str, default="p_{T}^{t,had} [GeV]")
parser.add_argument('--nrebin', '-nrebin', type=int, default=1)
parser.add_argument('--SplitFromBinLow', '-SplitFromBinLow', type=int, default=0)
parser.add_argument('--ParameterSplitFromBinLow', '-ParameterSplitFromBinLow', type=float, default=1.5)
parser.add_argument('--SplitFromBinHigh', '-SplitFromBinHigh', type=int, default=0)
parser.add_argument('--ParameterSplitFromBinHigh', '-ParameterSplitFromBinHigh', type=float, default=1.5)
parser.add_argument('--maxiterations', '-maxiterations', type=int, default=30)
parser.add_argument('--batch', '-batch', type=int, default=0)

args = parser.parse_args()

if (args.batch == 1):
    ROOT.gROOT.SetBatch(1)

def DivideBinWidth(h):
    for j in range(1, h.GetNbinsX()+1):
        h.SetBinContent(j,((h.GetBinContent(j))/(h.GetBinWidth(j))))
        h.SetBinError(j,((h.GetBinError(j))/(h.GetBinWidth(j))))

def PrintCan(can, outputname):
    outputname = outputname.split('/')
    can.Print('./pic/'+outputname[len(outputname)-1] + '.png')
    can.Print('./pic/'+outputname[len(outputname)-1] + '.pdf')

def MakeListFromHisto(hist):
    vals = []
    for i in range(1, hist.GetXaxis().GetNbins()+1):
        val = hist.GetBinContent(i)
        vals.append(val)
    return vals

def NormalizeResponse(h2, tag = '_migra'):
    migra = h2.Clone(h2.GetName() + tag)
    for i in range(1, h2.GetXaxis().GetNbins()+1):
        sum = 0.
        for j in range(1, h2.GetYaxis().GetNbins()+1):
            val = h2.GetBinContent(i,j)
            sum = sum + val
        if sum > 0.:
            for j in range(1, h2.GetYaxis().GetNbins()+1):
                migra.SetBinContent(i,j,migra.GetBinContent(i,j) / sum)
    return migra

def CheckNormalizeResponse(h2, tag = '_migra'):
    migra = h2.Clone(h2.GetName() + tag)
    for i in range(1, h2.GetYaxis().GetNbins()+1):
        sum = 0.
        for j in range(1, h2.GetXaxis().GetNbins()+1):
            val = h2.GetBinContent(j,i)
            sum = sum + val
        if sum > 0.:
            for j in range(1, h2.GetXaxis().GetNbins()+1):
                migra.SetBinContent(j,j,migra.GetBinContent(j,i) / sum)
    return migra

def MakeListResponse(h2):
    vals = []
    for i in range(1, h2.GetYaxis().GetNbins()+1):
        column = []
        for j in range(1, h2.GetXaxis().GetNbins()+1):
            val = h2.GetBinContent(i,j)
            column.append(val)
        vals.append(column)
    return vals

def MakeUnfoldedHisto(reco4bins, h1s, tag = '_unfolded'):
    hname = reco4bins.GetName()+tag
    hist = reco4bins.Clone(hname)
    hist.Reset()
    histMean = reco4bins.Clone(hname+"Mean")
    histMean.Reset()
    i = -1
    for h1 in h1s:
        i = i+1
        h1.Rebin(8)
        h1.Fit("gaus")
        fit = h1.GetFunction("gaus") 
        chi2 = fit.GetChisquare()
        p1 = fit.GetParameter(1)
        p0 = fit.GetParameter(0)
        p2 = fit.GetParameter(2)
        hist.SetBinContent(i+1, p1)
        hist.SetBinError(i+1, p2)
        histMean.SetBinContent(i+1, h1.GetMean())
        histMean.SetBinError(i+1, h1.GetRMS())
    return hist, histMean

def MakeTH1Ds(trace, tag = 'trace', nbins = 600):
    h1s = []
    gmax = -999
    gmin = 1e19
    for line in trace:
        xmin = min(line)
        xmax = max(line)
        gmin = min(xmin, gmin)
        gmax = max(xmax, gmax)
    i = -1
    gmax = -999
    gmin = 1e19
    
    for line in trace:
        gxmin = min(line)
        gxmax = max(line)
        i = i+1
        hname = tag + '_{:}'.format(i)
        h1 = TH1D(hname, hname, nbins, gmin, gmax)
        print("-----------------------")
        k=0
        for val in line:
            h1.Fill(val)
        h1s.append(h1)
        print(i)
        print("Appending traces")
        print(h1)
        histograms.append(h1)
    return h1s

def TransposeMatrix(h_response_unf):
    h_responce_transpose = h_response_unf.Clone(h_response_unf.GetName()+"clone")
    h_responce_transpose.Reset()
    for i in range(1,h_response_unf.GetXaxis().GetNbins()+1):
        for j in range(1,h_response_unf.GetXaxis().GetNbins()+1):
            h_responce_transpose.SetBinContent(i,j,h_response_unf.GetBinContent(j,i))
            h_responce_transpose.SetBinError(i,j,h_response_unf.GetBinError(j,i))
    h_responce_transpose.GetXaxis().SetTitle(h_response_unf.GetYaxis().GetTitle())
    h_responce_transpose.GetYaxis().SetTitle(h_response_unf.GetXaxis().GetTitle())
    return h_responce_transpose
    
def SaveHistograms(outputname = "output"):
    outputname = outputname.split('/')
    outfile = TFile('./outputs/'+outputname[len(outputname)-1]+".root", 'recreate')
    outfile.cd()
    for j in range(len(histograms)):
        histograms[j].Write()
    print("End of the unfolding.")
    outfile.Write()
    outfile.Close() 

def PlotPosteriors(ListOfPosteriors, outputname = ""):
    RepeatIteration = False
    if (outputname != "" ):
        outputname = "_"+outputname
    c = TCanvas("Posteriors"+outputname,"Posteriors"+outputname,0,0,1600, 1600)
    c.Divide(math.ceil(pow(len(ListOfPosteriors),0.5)),math.ceil(pow(len(ListOfPosteriors),0.5)))
    legends = []
    for i in range(len(ListOfPosteriors)):
        ListOfPosteriors[i].SetMaximum(ListOfPosteriors[i].GetMaximum()*2.0)
        #ListOfPosteriors[i].Fit("gaus")
        fit = ListOfPosteriors[i].GetFunction("gaus") 
        chi2 = fit.GetChisquare()
        p1 = fit.GetParameter(1)
        #p0 = fit.GetParameter(0)
        p2 = fit.GetParameter(2)
        FitIntegral = fit.Integral(p1-10*p2,p1+10*p2) # fit integral , get mean plus minus 10 sigma
        PriorIntegral = ListOfPosteriors[i].Integral("width")
        Percentage = round((100*PriorIntegral/FitIntegral),0)
        c.cd(i+1)
        gPad.SetFrameFillColor(10)
        if (Percentage < 90):
            gPad.SetFillColor(kRed-4)
            RepeatIteration = True
        else:
            gPad.SetFillColor(8)
        leg = TLegend(0.1,0.5,0.85,0.9)
        leg.SetFillStyle(0)
        leg.AddEntry(None,"MeanHist = "+str(round(ListOfPosteriors[i].GetMean(),0))+", RMShist = "+str(round(ListOfPosteriors[i].GetRMS(),0))+", MeanFit = "+str(round(p1,0))+", #sigma_{fit} = "+str(round(p2,0)),"")
        leg.AddEntry(ListOfPosteriors[i],"Posterior ","l")
        leg.AddEntry(fit,"Fit ","l")
        leg.AddEntry(None,"#chi^{2}/NDF = "+str(round(chi2/len(ListOfPosteriors),2)),"")
        leg.AddEntry(None,"Integral Prior/Fit = "+str(round(Percentage,0))+" %","")
        leg.SetBorderSize(0)
        legends.append(leg)
        ListOfPosteriors[i].SetTitle("Posterior in bin "+str(i+1))
        gStyle.SetOptStat(0)
        ListOfPosteriors[i].Draw()
        legends[i].Draw("same")
        if (round(chi2/len(ListOfPosteriors),2)) > 20.0:
            RepeatIteration = True
        #c.Update()

    c.cd(1)
    gStyle.SetOptStat(0)
    #gPad.Modified()
    #c.Update()
    PrintCan(c,c.GetName()+"_posteriors")
    return RepeatIteration



def PlotRatio(h_reco_unfolded,h_ptcl_or,h_reco_unfolded_roof, h_reco_unfolded_svd, h_reco_unfolded_T,h_reco_unfolded_Ids, svd_par, Ids_par ,outputname="test.png"):
    
    gStyle.SetPadLeftMargin(0.12)
    gStyle.SetPadRightMargin(0.12)
    c = TCanvas("canvas_"+outputname,"canvas_"+outputname,0,0,800, 800)
    c.cd()
    pad1 = TPad("pad1","pad1",0,0.40,1,1)
    pad1.SetTopMargin(0.15)
    pad1.SetBottomMargin(0.01)
    pad1.SetFillStyle(0)
    pad1.SetTicks(1,1)
    pad1.SetBorderMode(0)
    pad1.Draw()
    c.cd()
    pad2 = TPad("pad2","pad2",0,0.01,1,0.422)
    pad2.SetFillStyle(0)
    pad2.SetTopMargin(0.043)
    pad2.SetBottomMargin(0.2)
    pad2.SetBorderMode(0)
    pad2.SetTicks(1,1)
    pad2.Draw()
    pad2.Modified()
    c.cd()
    pad1.cd()
    gStyle.SetOptStat(0)
    h_ptcl_or.SetTitle("")
    h_ptcl_or.SetLineColor(2)
    #h_ptcl_or.GetYaxis().SetRangeUser(0,h_ptcl_or.GetMaximum()*1.5)
    h_ptcl_or.GetXaxis().SetTitleSize(34)
    h_ptcl_or.GetXaxis().SetTitleFont(43)
    h_ptcl_or.GetYaxis().SetTitleSize(27)
    h_ptcl_or.GetYaxis().SetTitleFont(43)
    h_ptcl_or.GetYaxis().SetTitleOffset(1.5)
    h_ptcl_or.GetYaxis().SetTitle("Events")
    h_ptcl_or.GetYaxis().SetLabelFont(43)
    h_ptcl_or.GetYaxis().SetLabelSize(25)
    legend = TLegend(0.55,0.5,0.85,0.8)
    legend.SetFillStyle(0)
    legend.AddEntry(h_ptcl_or,"Simulation")
    legend.AddEntry(h_reco_unfolded,"Fully Bayesian Unfolding","p")
    legend.AddEntry(h_reco_unfolded_roof,"D'Agostini RooUnfold, par. 4","p")
    legend.AddEntry(h_reco_unfolded_svd,"SVD RooUnfold, par. " + str(svd_par),"p")
    legend.AddEntry(h_reco_unfolded_T,"T RooUnfold","p")
    legend.AddEntry(h_reco_unfolded_Ids,"Ids RooUnfold, par. " + str(Ids_par),"p")
    legend.SetBorderSize(0)
    h_reco_unfolded.SetLineColor(1)
    h_reco_unfolded.SetMarkerColor(1)
    h_reco_unfolded.SetMarkerStyle(22)
    h_reco_unfolded_roof.SetMarkerColor(6)
    h_reco_unfolded_roof.SetLineColor(6)
    h_reco_unfolded_roof.SetMarkerStyle(20)
    h_reco_unfolded_svd.SetMarkerColor(4)
    h_reco_unfolded_svd.SetLineColor(4)
    h_reco_unfolded_svd.SetMarkerStyle(5)
    h_reco_unfolded_T.SetMarkerColor(7)
    h_reco_unfolded_T.SetLineColor(7)
    h_reco_unfolded_T.SetMarkerStyle(34)
    h_reco_unfolded_Ids.SetMarkerColor(8)
    h_reco_unfolded_Ids.SetLineColor(8)
    h_reco_unfolded_Ids.SetMarkerStyle(3)
    h_ptcl_or.Draw("hist")
    h_reco_unfolded.Draw("same p x0")
    h_reco_unfolded_roof.Draw("same p x0")
    h_reco_unfolded_svd.Draw("same p x0")
    h_reco_unfolded_T.Draw("same p x0")
    h_reco_unfolded_Ids.Draw("same p x0")
    legend.Draw("same")
    pad1.RedrawAxis()
    pad2.cd()
    h_ptcl_or_clone = h_ptcl_or.Clone(h_ptcl_or.GetName()+"_clone")
    h_reco_unfolded_clone = h_reco_unfolded.Clone(h_reco_unfolded.GetName()+"_clone")
    h_reco_unfolded_roof_clone = h_reco_unfolded_roof.Clone(h_reco_unfolded_roof.GetName()+"_clone")
    h_reco_unfolded_svd_clone = h_reco_unfolded_svd.Clone(h_reco_unfolded_svd.GetName()+"_clone")
    h_reco_unfolded_T_clone = h_reco_unfolded_T.Clone(h_reco_unfolded_T.GetName()+"_clone")
    h_reco_unfolded_Ids_clone = h_reco_unfolded_Ids.Clone(h_reco_unfolded_Ids.GetName()+"_clone")
    h_ptcl_or_clone.Divide(h_ptcl_or)
    h_reco_unfolded_clone.Divide(h_ptcl_or)
    h_reco_unfolded_roof_clone.Divide(h_ptcl_or)
    h_reco_unfolded_svd_clone.Divide(h_ptcl_or)
    h_reco_unfolded_T_clone.Divide(h_ptcl_or)
    h_reco_unfolded_Ids_clone.Divide(h_ptcl_or)
    
    h_ptcl_or_clone.GetXaxis().SetTitleSize(27)
    h_ptcl_or_clone.GetXaxis().SetTitleFont(43)
    h_ptcl_or_clone.GetYaxis().SetTitleSize(27)
    h_ptcl_or_clone.GetYaxis().SetTitleFont(43)
    
    h_ptcl_or_clone.GetXaxis().SetLabelFont(43)
    h_ptcl_or_clone.GetXaxis().SetLabelSize(25)
    h_ptcl_or_clone.GetYaxis().SetLabelFont(43)
    h_ptcl_or_clone.GetYaxis().SetLabelSize(25)
    
    h_ptcl_or_clone.SetMaximum(1.3)
    h_ptcl_or_clone.SetMinimum(0.7)
    
    h_ptcl_or_clone.GetXaxis().SetTitleOffset(2.5)
    h_ptcl_or_clone.GetXaxis().SetTitle(args.title)
    
    h_ptcl_or_clone.GetYaxis().SetTitle("#frac{Unfolded}{Simulation}      ")
    h_ptcl_or_clone.Draw("hist")
    h_reco_unfolded_clone.Draw("same p x0")
    h_reco_unfolded_roof_clone.Draw("same p x0")
    h_reco_unfolded_svd_clone.Draw("same p x0")
    h_reco_unfolded_T_clone.Draw("same p x0")
    h_reco_unfolded_Ids_clone.Draw("same p x0")
    pad2.RedrawAxis()
    c.Update()
    histograms.append(c) # here is the crash probably
    PrintCan(c, outputname)

def MyRooUnfold(matrix_name=args.h_matrix, h_reco_getG0_name=args.h_data, h_ptcl_getG0_name = args.h_particle,h_reco_get_bkg_name = args.h_background,outputname=args.h_data+"_unfolded",nrebin = args.nrebin):

    rfile_data = TFile(args.rfile_data, 'read')
    rfile_particle = TFile(args.rfile_particle, 'read')
    rfile_matrix = TFile(args.rfile_matrix, 'read')
    rfile_background = TFile(args.rfile_background, 'read')

    myfbu = fbu.PyFBU()
    myfbu.verbose = True 
    #GET DATA
    h_reco_get = rfile_data.Get(h_reco_getG0_name)
    h_reco_get.Rebin(nrebin)
    #GET PARTICLE
    h_ptcl_get = rfile_particle.Get(h_ptcl_getG0_name)
    h_ptcl_get.Rebin(nrebin)
    #GET MATRIX
    h_response_unf = rfile_matrix.Get(matrix_name)
    h_response_unf.ClearUnderflowAndOverflow()
    h_response_unf.GetXaxis().SetRange(1, h_response_unf.GetXaxis().GetNbins() )
    h_response_unf.GetYaxis().SetRange(1, h_response_unf.GetYaxis().GetNbins() )
    h_response_unf.Rebin2D(nrebin,nrebin)
    h_response_unf.SetName("Migration_Matrix_simulation")

    ########### ACCEPTANCY
    h_acc = h_response_unf.ProjectionX("reco_recoandparticleX") # Reco M
    h_acc.Divide(h_reco_get)
    ########### AKCEPTANCE saved in h_acc #############
    ########### EFFICIENCY
    h_eff = h_response_unf.ProjectionY("reco_recoandparticleY") # Ptcl M
    h_eff.Divide(h_ptcl_get)
    
    h_reco_get_input = rfile_data.Get(h_reco_getG0_name)
    h_reco_get_bkg = rfile_background.Get(h_reco_get_bkg_name)
    h_reco_get_bkg.Rebin(nrebin)

    h_reco_get_input_clone=h_reco_get_input.Clone("")

    h_reco_get_input_clone.Add(h_reco_get_bkg,-1)
    h_reco_get_input_clone.Multiply(h_acc)
    
   
    h_reco_or = rfile_data.Get(h_reco_getG0_name)
    h_ptcl_or = rfile_particle.Get(h_ptcl_getG0_name)
    h_ptcl_or.SetMaximum(h_ptcl_or.GetMaximum()*1.5)
    
    ### ROOUNFOLD METHOD ###
    
    m_RooUnfold = RooUnfoldBayes()
    m_RooUnfold.SetRegParm( 4 )
    m_RooUnfold.SetNToys( 10000 )
    m_RooUnfold.SetVerbose( 0 )
    m_RooUnfold.SetSmoothing( 0 )
  
    response = RooUnfoldResponse(None, None, h_response_unf, "response", "methods")
    
    m_RooUnfold.SetResponse( response )
    m_RooUnfold.SetMeasured( h_reco_get_input_clone )
    
    ### SVD METHOD ###
    
    m_RooUnfold_svd = RooUnfoldSvd (response, h_reco_get_input_clone, int(round(h_reco_get_input_clone.GetNbinsX()/2.0,0))) #8
    svd_par = int(round(h_reco_get_input_clone.GetNbinsX()/2.0,0))
    m_RooUnfold_T = RooUnfoldTUnfold (response, h_reco_get_input_clone)         #  OR
    m_RooUnfold_Ids= RooUnfoldIds (response, h_reco_get_input_clone,int(round(h_reco_get_input_clone.GetNbinsX()/12.0,0))) ## TO DO, SET PARAMETERS TO THE BINNING
    Ids_par = int(round(h_reco_get_input_clone.GetNbinsX()/12.0,0))
    
    ### FBU METHOD ###
    
    h_response_unf_fbu = TransposeMatrix(h_response_unf)
    h_response_unf_fbu_norm = NormalizeResponse(h_response_unf_fbu)
    h_response_unf_fbu_norm.SetName("Migration_Matrix_simulation_transpose")
    histograms.append(h_response_unf_fbu_norm)
    myfbu.response = MakeListResponse(h_response_unf_fbu_norm)
    myfbu.data = MakeListFromHisto(h_reco_get_input_clone) 
    myfbu.lower = []
    myfbu.upper = []
    
    h_det_div_ptcl=h_reco_get_input_clone.Clone("")
    h_det_div_ptcl.Divide(h_ptcl_or)
    h_det_div_ptcl.Divide(h_eff)
    h_det_div_ptcl.SetName("det_div_ptcl")
    histograms.append(h_det_div_ptcl)

    for l in range(len(myfbu.data)):
        if ( args.SplitFromBinLow != 0) and ( l+1 <= args.SplitFromBinLow ):
            myfbu.lower.append(h_reco_get_input_clone.GetBinContent(l+1)*(2-args.ParameterSplitFromBinLow)*h_det_div_ptcl.GetBinContent(l+1))
            myfbu.upper.append(h_reco_get_input_clone.GetBinContent(l+1)*args.ParameterSplitFromBinLow*h_det_div_ptcl.GetBinContent(l+1))
        elif ( args.SplitFromBinHigh != 0 ) and ( l+1 >= args.SplitFromBinHigh ):
            myfbu.lower.append(h_reco_get_input_clone.GetBinContent(l+1)*(2-args.ParameterSplitFromBinHigh)*h_det_div_ptcl.GetBinContent(l+1))
            myfbu.upper.append(h_reco_get_input_clone.GetBinContent(l+1)*args.ParameterSplitFromBinHigh*h_det_div_ptcl.GetBinContent(l+1))
        else:
            myfbu.lower.append(h_reco_get_input_clone.GetBinContent(l+1)*(2-args.par)*h_det_div_ptcl.GetBinContent(l+1))
            myfbu.upper.append(h_reco_get_input_clone.GetBinContent(l+1)*args.par*h_det_div_ptcl.GetBinContent(l+1))
    #myfbu.regularization = Regularization('Tikhonov',parameters=[{'refcurv':0.1,'alpha':0.2}]) works for old FBU package and python2.7 and old pymc
    myfbu.run()
    trace = myfbu.trace
    traceName = 'Posterior_1_iteration'
    posteriors_diag = MakeTH1Ds(trace, traceName)
    h_reco_unfolded, h_reco_unfolded_Mean = MakeUnfoldedHisto(h_reco_or, posteriors_diag)
    PlotPosteriors(posteriors_diag,outputname+'_iteration_1')
    # EFFICIENCY AND ACCEPTANCY CORRECTIONS
    h_reco_unfolded.Divide(h_eff)
    h_reco_unfolded_Mean.Divide(h_eff)

    h_reco_unfolded_roof = m_RooUnfold.Hreco()
    h_reco_unfolded_roof.Divide(h_eff)

    h_reco_unfolded_svd = m_RooUnfold_svd.Hreco()
    h_reco_unfolded_svd.Divide(h_eff)

    h_reco_unfolded_T = m_RooUnfold_T.Hreco()
    h_reco_unfolded_T.Divide(h_eff)

    h_reco_unfolded_Ids = m_RooUnfold_Ids.Hreco()
    h_reco_unfolded_Ids.Divide(h_eff)

    PlotRatio(h_reco_unfolded_Mean, h_ptcl_or, h_reco_unfolded_roof, h_reco_unfolded_svd, h_reco_unfolded_T,h_reco_unfolded_Ids, svd_par, Ids_par, outputname+'_iteration_1')        

    Repeat = True
    j = 2
    while Repeat:
        print("Runnig iteration number: ",j)
        myfbu.lower = []
        myfbu.upper = []
        for l in range(len(myfbu.data)):
            posteriors_diag[l].Fit("gaus")
            fit = posteriors_diag[l].GetFunction("gaus") 
            p1 = fit.GetParameter(1)
            p2 = fit.GetParameter(2)
            myfbu.lower.append(p1-4*p2)
            myfbu.upper.append(p1+4*p2)
        myfbu.run()
        trace = myfbu.trace
        traceName = 'Posterior_'+str(j)+'_iteration'
        posteriors_diag = MakeTH1Ds(trace, traceName)
        h_reco_unfolded, h_reco_unfolded_Mean = MakeUnfoldedHisto(h_reco_or, posteriors_diag)
        Repeat = PlotPosteriors(posteriors_diag,outputname+'_iteration_'+str(j))
        # EFFICIENCY AND ACCEPTANCY CORRECTIONS
        h_reco_unfolded.Divide(h_eff)
        h_reco_unfolded_Mean.Divide(h_eff)
        h_reco_unfolded_roof = m_RooUnfold.Hreco()
        h_reco_unfolded_roof.Divide(h_eff)
        h_reco_unfolded_svd = m_RooUnfold_svd.Hreco()
        h_reco_unfolded_svd.Divide(h_eff)
        h_reco_unfolded_T = m_RooUnfold_T.Hreco()
        h_reco_unfolded_T.Divide(h_eff)
        h_reco_unfolded_Ids = m_RooUnfold_Ids.Hreco()
        h_reco_unfolded_Ids.Divide(h_eff)
        PlotRatio(h_reco_unfolded_Mean, h_ptcl_or, h_reco_unfolded_roof, h_reco_unfolded_svd, h_reco_unfolded_T,h_reco_unfolded_Ids, svd_par, Ids_par, outputname+'_iteration_'+str(j))
        if j == args.maxiterations:
            break
        j = j+1

    h_reco_unfolded.SetName("result_fbu_fit")
    histograms.append(h_reco_unfolded)
    
    h_reco_unfolded_Mean.SetName("result_fbu_Mean")
    histograms.append(h_reco_unfolded_Mean)
    
    h_reco_unfolded_roof.SetName("result_roof")
    histograms.append(h_reco_unfolded_roof)
    
    h_reco_unfolded_svd.SetName("result_svd")
    histograms.append(h_reco_unfolded_svd)
    
    h_reco_unfolded_T.SetName("result_T")
    histograms.append(h_reco_unfolded_T)
    
    h_reco_unfolded_Ids.SetName("result_Ids")
    histograms.append(h_reco_unfolded_Ids)

    h_eff.SetName("efficiency")
    histograms.append(h_eff)
    h_acc.SetName("acceptancy")
    histograms.append(h_acc)
    
    h_reco_or.SetName("reco")
    histograms.append(h_reco_or)
    h_ptcl_or.SetName("ptcl_simulation")
    histograms.append(h_ptcl_or)

    h_ratio = h_reco_unfolded.Clone("")
    h_ratio.Divide(h_ptcl_or)
    h_ratio.SetName("ratio_fbu_fit")
    histograms.append(h_ratio)
    
    h_ratio = h_reco_unfolded_Mean.Clone("")
    h_ratio.Divide(h_ptcl_or)
    h_ratio.SetName("ratio_fbu_Mean")
    histograms.append(h_ratio)

    h_ratio = h_reco_unfolded_roof.Clone("")
    h_ratio.Divide(h_ptcl_or)
    h_ratio.SetName("ratio_roof")
    histograms.append(h_ratio)
    
    h_ratio = h_reco_unfolded_svd.Clone("")
    h_ratio.Divide(h_ptcl_or)
    h_ratio.SetName("ratio_svd")
    histograms.append(h_ratio)

    m_RooUnfold_svd.PrintTable (cout, h_ptcl_or)
    m_RooUnfold.PrintTable (cout, h_ptcl_or)
  
    # CORRECTIONS TO GET CROSS SECTION

    #DivideBinWidth(h_reco_unfolded_Mean)
    #DivideBinWidth(h_reco_unfolded_roof)
    #DivideBinWidth(h_reco_unfolded_svd)
    #DivideBinWidth(h_reco_unfolded_T)
    #DivideBinWidth(h_reco_unfolded_Ids)
    #DivideBinWidth(h_ptcl_or)
    #Lumi = 36.1e3
    #for j in range(1,h_reco_unfolded_Mean.GetXaxis().GetNbins()+1):
    #    h_reco_unfolded_Mean.SetBinContent(j,h_reco_unfolded_Mean.GetBinContent(j)/(Lumi))
    #    h_reco_unfolded_Mean.SetBinError(j,h_reco_unfolded_Mean.GetBinError(j)/(Lumi))
    #    h_reco_unfolded_roof.SetBinContent(j,h_reco_unfolded_roof.GetBinContent(j)/(Lumi))
    #    h_reco_unfolded_roof.SetBinError(j,h_reco_unfolded_roof.GetBinError(j)/(Lumi))
    #    h_reco_unfolded_svd.SetBinContent(j,h_reco_unfolded_svd.GetBinContent(j)/(Lumi))
    #    h_reco_unfolded_svd.SetBinError(j,h_reco_unfolded_svd.GetBinError(j)/(Lumi))
    #    h_reco_unfolded_T.SetBinContent(j,h_reco_unfolded_T.GetBinContent(j)/(Lumi))
    #    h_reco_unfolded_T.SetBinError(j,h_reco_unfolded_T.GetBinError(j)/(Lumi))
    #    h_reco_unfolded_Ids.SetBinContent(j,h_reco_unfolded_Ids.GetBinContent(j)/(Lumi))
    #    h_reco_unfolded_Ids.SetBinError(j,h_reco_unfolded_Ids.GetBinError(j)/(Lumi))
    #    h_ptcl_or.SetBinContent(j,h_ptcl_or.GetBinContent(j)/(Lumi))
    #    h_ptcl_or.SetBinError(j,h_ptcl_or.GetBinError(j)/(Lumi))
    #h_reco_unfolded_Mean_clone=h_reco_unfolded_Mean.Clone("FBU_cross_section")
    #h_reco_unfolded_roof_clone=h_reco_unfolded_roof.Clone("DAgostini_cross_section")
    #h_reco_unfolded_svd_clone=h_reco_unfolded_svd.Clone("SVD_cross_section")
    #h_reco_unfolded_T_clone=h_reco_unfolded_T.Clone("TUnfold_cross_section")
    #h_reco_unfolded_Ids_clone=h_reco_unfolded_Ids.Clone("Ids_cross_section")
    #h_ptcl_or_clone=h_ptcl_or.Clone("Truth_cross_section")
    #
    #print("CONTROL*******************************************************************: ",h_reco_unfolded_Mean_clone.GetXaxis().GetNbins(),h_reco_unfolded_roof_clone.GetXaxis().GetNbins(),h_reco_unfolded_svd_clone.GetXaxis().GetNbins(),h_reco_unfolded_T_clone.GetXaxis().GetNbins(),h_reco_unfolded_Ids_clone.GetXaxis().GetNbins(),h_ptcl_or_clone.GetXaxis().GetNbins())
    #histograms.append(h_reco_unfolded_Mean_clone)
    #histograms.append(h_reco_unfolded_roof_clone)
    #histograms.append(h_reco_unfolded_svd_clone)
    #histograms.append(h_reco_unfolded_T_clone)
    #histograms.append(h_reco_unfolded_Ids_clone)
    #histograms.append(h_ptcl_or_clone)
    SaveHistograms(outputname)
    #PlotRatio(h_reco_unfolded_Mean, h_ptcl_or, h_reco_unfolded_roof, h_reco_unfolded_svd, h_reco_unfolded_T,h_reco_unfolded_Ids, svd_par, Ids_par, outputname)

histograms = []

MyRooUnfold()