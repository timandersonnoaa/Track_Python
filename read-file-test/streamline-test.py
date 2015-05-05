
"""
Created on Thu Jan  8 10:18:40 2015

@author: bling
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from track_functions import get_drifter,get_fvcom,get_roms,draw_basemap,distance,uniquecolors
from matplotlib import animation
from pandas import Series,DataFrame
st_run_time = datetime.now()

MODEL = 'FVCOM'
image_style = 'animation' # or 'animation'
start_time = datetime(2013,4,15,0,0,0,0,pytz.UTC)
stp_num = 1; seg_num = 49
model_days = 1; depth = -1
lon_set = [[]]*stp_num; lat_set = [[]]*stp_num
loop_length = []; 
streamline = 'ON'
points = {'lats':[],'lons':[]}

############### read ##############
point=np.load('model_points.npz')
lon_set = point['lon']; lat_set = point['lat']

for i in xrange(stp_num):
    loop_length.append(len(lon_set[i]))
    points['lats'].extend(lat_set[i]); 
    points['lons'].extend(lon_set[i])#'''

model_points=np.load('streamline.npz')
lonpps=model_points['lonpps']; latpps=model_points['latpps'];
US=model_points['US']; VS=model_points['VS']; speeds=model_points['speeds']

'''for i in xrange(stp_num):
    loop_length.append(len(lon_set[i]))
    points['lats'].extend(lat_set[i]); 
    points['lons'].extend(lon_set[i])#'''
# Get elements boundary
'''b_points = np.load('boundary-points.npz')
blon = b_points['lon']; blat = b_points['lat']; 
points['lons'].extend(blon); points['lats'].extend(blat);#'''

# Get nodes boundary
'''nodes_points = np.load('nodes_points.npz')
nlon = nodes_points['lon']; nlat = nodes_points['lat']
points['lons'].extend(nlon); points['lats'].extend(nlat)#'''
'''nlons = []; nlats = []
for i in xrange(len(nlon)):
    nlons.extend(nlon[i]); nlats.extend(nlat[i])
    points['lons'].extend(nlon[i]); points['lats'].extend(nlat[i]);
np.savez('nodes_point.npz',lon=nlons,lat=nlats)#'''

############## plot ##############
fig = plt.figure() #figsize=(23,15)
ax = fig.add_subplot(111)
plt.suptitle('FVCOM: massbay model,forecast 1 day')
#draw_basemap(ax, points)

#%(stp_num,start_time.strftime('%D-%H:%M')))  # %m/%d-%H:%M
#plt.title('%.f%% simulated drifters ashore\n%d days, %d m, %s'%(int(round(68)),model_days,depth,start_time.strftime("%d-%b-%Y")))
#plt.suptitle('this is the figure title', fontsize=12)
#colors = uniquecolors(stp_num) #config colors
if image_style=='plot':
    #ax.plot(blon,blat,'bo',markersize=3)
    ax.plot(lon_back[0],lat_back[0],'bo-',markersize=4,label='backward')
    ax.plot(lon_set[0],lat_set[0],'ro-',markersize=4,label='forward')
    ax.annotate('Sighting 4/12/15', xy=(lon_set[0][0],lat_set[0][0]),xytext=(lon_set[0][0]+0.01,lat_set[0][0]+0.01),fontsize=12,arrowprops=dict(arrowstyle="fancy"))
    """for j in xrange(stp_num):
        '''ax.annotate('Start %d'%(j+1), xy=(lon_set[j][0],lat_set[j][0]),xytext=(lon_set[j][0]+0.03,lat_set[j][0]+0.03),
                    fontsize=6,arrowprops=dict(arrowstyle="wedge")) #facecolor=colors[i]'''
        ax.plot(lon_set[j][-1],lat_set[j][-1],'o',color=colors[j],markersize=4) #markerfacecolor='r',"""
       
if image_style=='animation':
    #ls = [[]]*stp_num
    
    def animate(n):#del ax.collections[:]; del ax.lines[:]; ax.cla();ax.clf()
        #del ax.lines[:]
        ax.cla()
        #del ax.collections[:]
        
        if streamline == 'ON':
            plt.streamplot(lonpps[n],latpps[n],US[n],VS[n], color=speeds[n],arrowsize=4,cmap=plt.cm.cool,density=2.0)
            
        for j in range(stp_num):
            #plt.title('FVCOM: backward 16 hours, forward 5 days')
            if n==0:#facecolor=colors[i]'''
                ax.annotate('Start %d'%(j+1), xy=(lon_set[j][0],lat_set[j][0]),xytext=(lon_set[j][0]+0.01*stp_num,
                            lat_set[j][0]+0.01*stp_num),fontsize=6,arrowprops=dict(arrowstyle="fancy")) 
            if n<len(lon_set[j]): #markerfacecolor='r',
                ax.plot(lon_set[j][:n+1],lat_set[j][:n+1],'o-',color='red',markersize=4,label='Start %d'%(j+1))
        draw_basemap(ax, points)
    anim = animation.FuncAnimation(fig, animate, frames=max(loop_length))#, interval=1000
if streamline == 'ON':
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
#plt.legend(loc='lower right',fontsize=12)
###################################################
en_run_time = datetime.now()
print 'Take '+str(en_run_time-st_run_time)+' run the code. End at '+str(en_run_time) 
if image_style=='plot':
    plt.savefig('%s-forecast_%s'%(MODEL,en_run_time.strftime("%d-%b-%H:%M")),dpi=400,bbox_inches='tight') #png_dir+
if image_style=='animation': #ffmpeg,imagemagick,mencoder fps=20'''
    anim.save('%s-forecast_%s.gif'%(MODEL,en_run_time.strftime("%d-%b-%H:%M")),writer='imagemagick',fps=1,dpi=250) #
plt.show()

