#!/usr/bin/env python
import logging,json,argparse,glob,sys,numpy as np
import ROOT
logger = logging.getLogger(__name__)
ROOT.gStyle.SetOptStat(0)
# ROOT.gROOT.SetBatch(1)

IMG_W = 5761
ETAMAX = 3.0
IMG_H = 256
DETA=0.025/24
DPHI=2.*np.pi/IMG_H
JET_SIZE = 0.4

pname = {
0: 'u/d',
1: 's',
2: 'c',
3: 'b',
4: 'gl',
5: 'e',
6: 'mu',
7: 'tau',
8: 'ph'
}

def main():
   global IMG_W,ETAMAX,IMG_H,DETA,DPHI
   parser = argparse.ArgumentParser(description='Atlas Training')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.')
   parser.add_argument('--input', '-i',
                       help='configuration in standard json format.')
   parser.add_argument('--etamax','-e',help='eta max cutoff',required=True,type=float)
   args = parser.parse_args()

   logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s')
   
   # load configuration
   if args.config_file:
      config_file = json.load(open(args.config_file))
      filelist = sorted(glob.glob(config_file['data_handling']['input_file_glob']))
   elif args.input:
      filelist = sorted(glob.glob(args.input))
   else:
      args.print_help()
      sys.exit(-1)


   logger.info('found %s files',len(filelist))

   # get the image shape:
   tmpdata = np.load(filelist[0])
   raw_image_shape = tmpdata['raw'].shape
   truth_shape = tmpdata['truth'].shape

   logger.info('image shape: %s    truth shape: %s',raw_image_shape,truth_shape)

   ETAMAX = args.etamax
   IMG_W = raw_image_shape[3]
   IMG_H = raw_image_shape[2]
   DETA = float(IMG_W)/2./ETAMAX
   DPHI = float(IMG_H)/2./np.pi
   # IMG_D = raw_image_shape[1]

   # PIX_PER_ETA = int(IMG_W / (2. * ETAMAX))
   # JET_PIX_ETA = PIX_PER_ETA * JET_SIZE
   # PIX_PER_PHI = int(IMG_H / (2. * np.pi))
   # JET_PIX_PHI = PIX_PER_PHI * JET_SIZE
   
   can = ROOT.TCanvas('can','can',0,0,800,600)
   # canB = ROOT.TCanvas('canB','canB',0,0,800,600)
   can.Divide(2,2)

   # h2d = ROOT.TH2D('h2d',';eta;phi',int(JET_PIX_ETA),0.,JET_PIX_ETA*1.,int(JET_PIX_PHI),0.,JET_PIX_PHI*1.)
   
   for i in range(len(filelist)):
      filename = filelist[i]
      logger.info('plotting file: %s',filename)
      logger.info('plotting file index: %s of %s',i,len(filelist))
      data = np.load(filename)
      images = data['raw']
      logger.info('image shape: %s',images.shape)
      batch_size,nchannels,bins_phi,bins_eta = images.shape
      truth  = data['truth']

      plot_3d(can,images,truth)
      
      
      
def plot_dist():
      for j in range(len(truth)):
         event = truth[j]
         image = images[j]

         hx = ROOT.TH1D('hx',';eta',IMG_W/5,0,IMG_W)
         hy = ROOT.TH1D('hy',';phi',IMG_H,0,IMG_H)
         hw = ROOT.TH1D('hw',';width',IMG_W/5,0,IMG_W)
         hh = ROOT.TH1D('hh',';height',IMG_H,0,IMG_H)


         for entry in event:
            x = entry[1]
            y = entry[2]
            if IMG_W/2 - 50 > x or x > IMG_W/2 + 50:
                continue
            h2d_each = ROOT.TH2D('h2d_each',';eta;phi',int(JET_PIX_ETA),0.,JET_PIX_ETA*1.,int(JET_PIX_PHI),0.,JET_PIX_PHI*1.)
            w = entry[3]
            h = entry[4]
            hx.Fill(x)
            hy.Fill(y)
            hw.Fill(w)
            hh.Fill(h)


            eta_min = int(x - JET_PIX_ETA/2)
            if eta_min < 0:
               eta_min = 0
            eta_max = int(x + JET_PIX_ETA/2)
            if eta_max >= IMG_W:
               eta_max = IMG_W - 1
            phi_min = int(y - JET_PIX_PHI/2)
            phi_max = int(y + JET_PIX_PHI/2)
            for channel in range(image.shape[0]):
               for phi in range(phi_min,phi_max):
                  bin_phi = phi - phi_min
                  data_phi = phi
                  if phi < 0:
                     data_phi = phi + IMG_H
                  elif phi >= IMG_H:
                     data_phi = phi % IMG_H
                  for eta in range(eta_min,eta_max):

                     if eta < 0 or eta >= IMG_W:
                        pass

                     h2d.Fill(eta-eta_min,bin_phi,image[channel][data_phi][eta])
                     h2d_each.Fill(eta-eta_min,bin_phi,image[channel][data_phi][eta])
        
         canB.cd()
         h2d_each.Draw('colz')
         canB.SaveAs('jetcone_each_%010d.png' % counter)

         can.cd(1)
         hx.Draw('hist')
         can.cd(2)
         hy.Draw('hist')
         can.cd(3)
         hw.Draw('hist')
         can.cd(4)
         hh.Draw('hist')
         can.Update()

         canB.cd()
         h2d.Draw('colz')
         canB.Update()
         canB.SaveAs('jetcone_%010d.png' % counter)
         counter += 1

         #can.SaveAs('dist.png')
         #canB.SaveAs('2d.png')
         raw_input('...')
   



def plot_3d(can,images,truth):
      batch_size,nchannels,bins_phi,bins_eta = images.shape

      for i in range(batch_size):
         logger.info('     index: %s',i)
         image = images[i]
         truth_list = truth[i]
         logger.info('image shape: %s      truth shape: %s',image.shape,truth_list.shape)
         logger.info('truth data: \n%s',truth_list)
         logger.info(' min = %s; max = %s',np.min(image),np.max(image))

         image[image < 0.25] = 0

         eta_phi_sum = np.sum(image,0)
         eta_depth_sum = np.sum(image,1)
         phi_depth_sum = np.sum(image,2)
         # xx = np.transpose(xx)
         # logger.info('xx = %s',xx.shape)

         h2_eta_phi = ROOT.TH2D('h2_eta_phi',';eta;phi',bins_eta,-1.*ETAMAX,ETAMAX,bins_phi,-np.pi,np.pi)
         h2_eta_phi.SetMinimum(0)
         h2_eta_depth = ROOT.TH2D('h2_eta_depth',';eta;depth',bins_eta,-1.*ETAMAX,ETAMAX,nchannels,0,nchannels)
         h2_eta_depth.SetMinimum(0)
         h2_phi_depth = ROOT.TH2D('h2_phi_depth',';phi;depth',bins_phi,-np.pi,np.pi,nchannels,0,nchannels)
         h2_phi_depth.SetMinimum(0)


         for depth in xrange(8,16):  # image.shape[0]):

            for phi in xrange(image.shape[1]):

               for eta in xrange(image.shape[2]):

                  tmp_eta = float(eta)/DETA - ETAMAX
                  tmp_phi = float(phi)/DPHI - np.pi

                  # logger.info('filling %s -> %s  %s -> %s',eta,tmp_eta,phi,tmp_phi)

                  h2_eta_phi.Fill(tmp_eta,tmp_phi,eta_phi_sum[phi][eta])
                  #h2_eta_depth.Fill(tmp_eta,depth,eta_depth_sum[depth][eta])
                  #h2_phi_depth.Fill(tmp_phi,depth,phi_depth_sum[depth][phi])


         can.cd()
         # can.cd(1).SetLogz()
         h2_eta_phi.Draw('colz')

         # logger.info('box = %s',y)
         boxes = []
         texts = []
         for y in truth_list:
            left = (y[1] - y[3] / 2)/DETA - ETAMAX
            bottom = (y[2] - y[4] / 2)/DPHI - np.pi
            right = (y[1] + y[3] / 2)/DETA - ETAMAX
            top = (y[2] + y[4] / 2)/DPHI - np.pi
            logger.debug(' (%s,%s) %s %s',left,bottom,right,top)

            box = ROOT.TBox(left,bottom,right,top)
            box.SetFillStyle(0)
            box.SetLineColor(y[5:].nonzero()[0][0]+1)
            box.SetLineWidth(1)
            box.Draw('same')
            boxes.append(box)

            text = ROOT.TPaveText(left,bottom,right,top)
            text.SetFillStyle(0)
            text.SetLineWidth(0)
            text.SetTextColor(ROOT.kBlack)
            #text.AddText(pname[np.argmax(y[5:])])
            text.SetTextSize(0.1)
            text.Draw('same')
            texts.append(text)


         # can.cd(2)
         # # can.cd(2).SetLogz()
         # h2_eta_depth.Draw('colz')

         # # logger.info('box = %s',y)
         # for y in truth_list:
         #    left = (y[1] - y[3] / 2)
         #    bottom = 0
         #    right = (y[1] + y[3] / 2)
         #    top = 16
         #    logger.debug(' (%s,%s) %s %s',left,bottom,right,top)

         #    box = ROOT.TBox(left,bottom,right,top)
         #    box.SetFillStyle(0)
         #    box.SetLineColor(y[5:].nonzero()[0][0]+1)
         #    box.SetLineWidth(1)
         #    box.Draw('same')
         #    boxes.append(box)

         # can.cd(3)
         # # can.cd(3).SetLogz()
         # h2_phi_depth.Draw('colz')

         # # logger.info('box = %s',y)
         # for y in truth_list:
         #    left = (y[2] - y[4] / 2)
         #    bottom = 0
         #    right = (y[2] + y[4] / 2)
         #    top = 16
         #    logger.debug(' (%s,%s) %s %s',left,bottom,right,top)

         #    box = ROOT.TBox(left,bottom,right,top)
         #    box.SetFillStyle(0)
         #    box.SetLineColor(y[5:].nonzero()[0][0]+1)
         #    box.SetLineWidth(1)
         #    box.Draw('same')
         #    boxes.append(box)

         can.Update()
         raw_input('...')


if __name__ == "__main__":
   print('start main')
   main()
