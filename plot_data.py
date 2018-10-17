#!/usr/bin/env python
import logging,json,argparse,glob,numpy as np
import ROOT
logger = logging.getLogger(__name__)


def main():
   parser = argparse.ArgumentParser(description='Atlas Training')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.')
   args = parser.parse_args()

   logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s')
   
   # load configuration
   config_file = json.load(open(args.config_file))
   filelist = sorted(glob.glob(config_file['data_handling']['input_file_glob']))
   
   can = ROOT.TCanvas('can','can',0,0,800,600)
   can.Divide(2,2)


   for filename in filelist:
      logger.info('plotting file: %s',filename)
      data = np.load(filename)
      images = data['raw']
      batch_size,nchannels,bins_phi,bins_eta = images.shape
      truth  = data['truth']


      for i in range(images.shape[0]):
         logger.info('     index: %s',i)
         image = images[i]
         y = truth[i][0]
         logger.info('%s %s',image.shape,y.shape)
         logger.info('%s',y)

         eta_phi_sum = np.sum(image,0)
         eta_depth_sum = np.sum(image,1)
         phi_depth_sum = np.sum(image,2)
         # xx = np.transpose(xx)
         # logger.info('xx = %s',xx.shape)

         h2_eta_phi = ROOT.TH2D('h2_eta_phi',';eta;phi',bins_eta,0,bins_eta,bins_phi,0,bins_phi)
         h2_eta_depth = ROOT.TH2D('h2_eta_depth',';eta;depth',bins_eta,0,bins_eta,nchannels,0,nchannels)
         h2_phi_depth = ROOT.TH2D('h2_phi_depth',';phi;depth',bins_phi,0,bins_phi,nchannels,0,nchannels)


         for depth in xrange(image.shape[0]):

            for phi in xrange(image.shape[1]):

               for eta in xrange(image.shape[2]):

                  h2_eta_phi.Fill(eta,phi,eta_phi_sum[phi][eta])
                  h2_eta_depth.Fill(eta,depth,eta_depth_sum[depth][eta])
                  h2_phi_depth.Fill(phi,depth,phi_depth_sum[depth][phi])


         can.cd(1)
         h2_eta_phi.Draw('colz')

         #logger.info('box = %s',y)
         left = (y[1] - y[3] / 2)
         bottom = (y[2] - y[4] / 2)
         right = (y[1] + y[3] / 2)
         top = (y[2] + y[4] / 2)
         logger.debug(' (%s,%s) %s %s',left,bottom,right,top)

         boxA = ROOT.TBox(left,bottom,right,top)
         boxA.SetFillStyle(0)
         boxA.SetLineColor(ROOT.kRed)
         boxA.SetLineWidth(1)
         boxA.Draw('same')


         can.cd(2)
         h2_eta_depth.Draw('colz')

         #logger.info('box = %s',y)
         left = (y[1] - y[3] / 2)
         bottom = 0
         right = (y[1] + y[3] / 2)
         top = 16
         logger.debug(' (%s,%s) %s %s',left,bottom,right,top)

         boxB = ROOT.TBox(left,bottom,right,top)
         boxB.SetFillStyle(0)
         boxB.SetLineColor(ROOT.kRed)
         boxB.SetLineWidth(1)
         boxB.Draw('same')

         can.cd(3)
         h2_phi_depth.Draw('colz')

         #logger.info('box = %s',y)
         left = (y[2] - y[4] / 2)
         bottom = 0
         right = (y[2] + y[4] / 2)
         top = 16
         logger.debug(' (%s,%s) %s %s',left,bottom,right,top)

         boxC = ROOT.TBox(left,bottom,right,top)
         boxC.SetFillStyle(0)
         boxC.SetLineColor(ROOT.kRed)
         boxC.SetLineWidth(1)
         boxC.Draw('same')

         can.Update()



         raw_input('...')



if __name__ == "__main__":
   print('start main')
   main()
