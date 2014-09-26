
import sys
import numpy as np
import random as rd

def getData(files):
  
  if len(files) < 1:
    print "No input file provided."
    exit()
  quas = False  
  if len(files) > 1:
    quas = True
    data_red = [line for line in open(files[1])]  
    arr_red = [float((data_red[i].split())[0]) for i in range(0,len(data_red))]
  
  data  = [line for line in open(files[0])]    
  arr_u = [float((data[i].split())[0]) for i in range(0,len(data))]
  arr_g = [float((data[i].split())[1]) for i in range(0,len(data))]
  arr_r = [float((data[i].split())[2]) for i in range(0,len(data))]
  arr_i = [float((data[i].split())[3]) for i in range(0,len(data))]
  arr_z = [float((data[i].split())[4]) for i in range(0,len(data))]

  obj_u = []; obj_g = []; obj_r = []; obj_i = []; obj_z = []; obj_red = []
  for i in range(0,len(data)):
    if (arr_u[i] > 0.):       
      obj_u.append(arr_u[i])
      obj_g.append(arr_g[i])
      obj_r.append(arr_r[i])
      obj_i.append(arr_i[i])
      obj_z.append(arr_z[i])
      if quas: obj_red.append(arr_red[i])
      else:    obj_red.append(0.)

  if quas:  return obj_u,obj_g,obj_r,obj_i,obj_z,obj_red
  else:     return obj_u,obj_g,obj_r,obj_i,obj_z

def main():

  trainingRatio = 0.7
    
  qu_u,qu_g,qu_r,qu_i,qu_z,qu_red = getData(["sdss_quasar_catalog.dat", "sdss_quasar_catalog_z.dat"])
  wd_u,wd_g,wd_r,wd_i,wd_z = getData(["sdss_white_dwarf_catalog.dat"])
  st_u,st_g,st_r,st_i,st_z = getData(["sdss_stars_catalog.dat"])
  
  quLen = len(qu_u)
  wdLen = len(wd_u)
  stLen = len(st_u)

  len1 = int(quLen); len2 = int(wdLen); len3 = int(stLen)
  shufSet = np.mat(np.zeros((len1+len2+len3,7)))

  sh = [i for i in range(len1)]
  rd.shuffle(sh)    
  for i in range(0,len1):
    idx = sh[i]
    shufSet[i,0] = qu_u[idx]; shufSet[i,1] = qu_g[idx]; shufSet[i,2] = qu_r[idx]; shufSet[i,3] = qu_i[idx]; shufSet[i,4] = qu_z[idx]; shufSet[i,5] = qu_red[idx]
    shufSet[i,6] = 0
  
  sh = [i for i in range(len2)]
  rd.shuffle(sh)
  for i in range(0,len2):
    idx = sh[i]
    shufSet[len1+i,0] = wd_u[idx]; shufSet[len1+i,1] = wd_g[idx]; shufSet[len1+i,2] = wd_r[idx]; shufSet[len1+i,3] = wd_i[idx]; shufSet[len1+i,4] = wd_z[idx]; shufSet[len1+i,5] = 0.
    shufSet[len1+i,6] = 1

  sh = [i for i in range(len3)]
  rd.shuffle(sh)
  for i in range(0,len3):
    idx = sh[i]
    shufSet[len1+len2+i,0] = st_u[idx]; shufSet[len1+len2+i,1] = st_g[idx]; shufSet[len1+len2+i,2] = st_r[idx]; shufSet[len1+len2+i,3] = st_i[idx]; shufSet[len1+len2+i,4] = st_z[idx]; shufSet[len1+len2+i,5] = 0.
    shufSet[len1+len2+i,6] = 2
  
  shuf2 = [i for i in range(len1+len2+len3)]
  rd.shuffle(shuf2)
  marker = int(trainingRatio*len(shufSet))
  f = open("trainA.dat", "w")
  for i in range(0,marker):  
    idx = shuf2[i]
    f.write(str(shufSet[idx,0]) + " " + str(shufSet[idx,1]) + " " + str(shufSet[idx,2]) + " " + str(shufSet[idx,3]) + " " + str(shufSet[idx,4]) + " " + str(shufSet[idx,5]) + " " + str(shufSet[idx,6]) + "\n")
  f.close()
  f = open("testA.dat", "w")
  for i in range(marker,len(shufSet)):  
    idx = shuf2[i]  
    f.write(str(shufSet[idx,0]) + " " + str(shufSet[idx,1]) + " " + str(shufSet[idx,2]) + " " + str(shufSet[idx,3]) + " " + str(shufSet[idx,4]) + " " + str(shufSet[idx,5]) + " " + str(shufSet[idx,6]) + "\n")
  f.close()
    
      
if __name__ == "__main__":
  main()
