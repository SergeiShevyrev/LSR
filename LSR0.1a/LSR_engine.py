# -*- coding: utf-8 -*-

#some methods are from https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
#this script has been composed and written by Sergei L Shevirev http://lefa.geologov.net

import gdal,ogr #OpenGIS Simple Features Reference Implementation
import numpy as np
from mygdal_functions0_9 import *

import matplotlib.pyplot as plt
import os
import pandas as pd
import time;

from skimage.transform import resize #function for SRTM resize according to Landsat

import elevation
import richdem as rd

import copy
"""
Landsat 8 OLI metadata page https://earthexplorer.usgs.gov/metadata/13400/LC81040292017298LGN00/
"""
def engineTopo(in_path='None',out_path='None',shpfilepath='None',drm_filepath='None',\
               products=[],bandStacks=[],is_topocorrection=False,SunElevation=30,\
               SunAzimuth=180,fileext="tif"):
    time_start=time.time()
    
    #some sample parameters for the topocorrection
    #is_topocorrection=True; #topocorrection flag
    #SunElevation=28.41189977  #31.23944509
    #SunAzimuth=163.93705102      #163.133415
    
    #Sun Elevation L1	 31.73425917
    #Sun Azimuth L1	162.99110733
    
    """
    #for 105-029
    #SunElevation=31.23944509 
    #SunAzimuth=163.133415      
    
    """
    if(drm_filepath=='None'):
        is_topocorrection=False;    
    
    SolarZenith=90-SunElevation;
    
    """
    You may exclude files from processing by renaming like ".tiff"
    """
        
    #files for processing, input and output directory
    #pathrowfolder="104_029"
    #datefolder="2015_11_05"
    #imgfilepath=os.path.join("..","Landsat8",pathrowfolder,datefolder); 
    #shpfilepath=os.path.join("..","shp",pathrowfolder+".shp");
    #shpfilepath=os.path.join("..","shp","AOI_tmp"+".shp");
    
    #fileext="tif"; #extention for files
    #outdir=os.path.join("..","Landsat8_Processed",pathrowfolder,datefolder);
    dir_cropped="cropped_bands_topo" #dir for AOI cropped
    dir_crop_path=os.path.join(out_path,dir_cropped);
    dir_products="products_topo" 
    dir_products_path=os.path.join(out_path,dir_products);
    band_number_inname='_b%N%.' #%N% - for band number e.g. LC81050292016143LGN00_B6.TIF NOT A CASE SENSITIVE
    band_number_inname=band_number_inname.lower();
    excl_pfix='_b8'; #endfile postfix to exclude from processing
    
    #drm for topocorrection
    #drm_name="mosaic_dem_south_kuril_utm.tif";
    #drm_folder=os.path.join("..","SRTM","files_for_mosaic");
    #drm_filepath=os.path.join(drm_folder,drm_name);
    
    #nodata srtm -32768
    
    #check is file/folder exists
    #print(os.path.isdir("/home/el"))
    #print(os.path.exists("/home/el/myfile.txt"))
    #

    file_for_crop=[];
    
    try:
        for file in os.listdir(in_path):
            #file=file.lower();
            if file.lower().endswith("."+fileext.lower()) and (file.lower().endswith(excl_pfix+'.'+fileext.lower()))==False:
                file_for_crop.append(file);
                print(file+" was added to crop queue.");
    except(FileNotFoundError):
            print("Input image folder doesn\'t exist...");
    """
    ДОПОЛНЕНИЯ в GUI сделать генерацию AOI shp выделением на изображении, если пользователь не генерирует 
    AOI, то AOI задать по размеру изображения
    """
    #STEP 0. Prepare for the topocorrection
    
    try:
        shp_extent=get_shp_extent(shpfilepath);
    except:
        print("Can not read shp AOI file. Applying extent from geotiff");
        gdal_object_tmp = gdal.Open(os.path.join(in_path,file_for_crop[0]))
        tmp_xsize = gdal_object_tmp.RasterXSize
        tmp_ysize = gdal_object_tmp.RasterYSize #x and y raster size in pixels
        tmp_gt=gdal_object_tmp.GetGeoTransform()
        tmp_ext=GetExtent(tmp_gt,tmp_ysize,tmp_xsize)
        shp_extent=[tmp_ext[0][0],tmp_ext[2][0],tmp_ext[1][1],tmp_ext[3][1]];
        
        
    #crop dem file
    if is_topocorrection==True:
        print("Perform cropping of srtm");
        
        #read DEM geotiff
        srtm_gdal_object = gdal.Open(drm_filepath)
        srtm_band = srtm_gdal_object.GetRasterBand(1)
        srtm_band_array = srtm_band.ReadAsArray() 
        
        #get spatial resolution
        srtm_gt=srtm_gdal_object.GetGeoTransform()
        srtm_xsize = srtm_gdal_object.RasterXSize
        srtm_ysize = srtm_gdal_object.RasterYSize #x and y raster size in pixels
        srtm_ext=GetExtent(srtm_gt,srtm_ysize,srtm_xsize) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
        #resolution in meters
        srtm_dpx=(srtm_ext[3][0]-srtm_ext[0][0])/srtm_xsize
        srtm_dpy=(srtm_ext[0][1]-srtm_ext[2][1])/srtm_ysize
        """
        print("srtm_ext={}".format(srtm_ext))
        print("shp_extent={}".format(shp_extent))
        """
        if check_shp_inside_raster(srtm_ext,shp_extent):
    #        sampleSrtmImage,ColMinIndSRTM,RowMinIndSRTM =crop_by_shp(shp_extent,srtm_ext,\
    #                                                    srtm_dpx,srtm_dpy,srtm_band_array); 
            srtm_band = rd.LoadGDAL(drm_filepath);
    
            slope = rd.TerrainAttribute(srtm_band, attrib='slope_degrees')
            aspect = rd.TerrainAttribute(srtm_band, attrib='aspect')
    
            rd.SaveGDAL(os.path.join(os.path.dirname(drm_filepath),"aspectInitialRes.tif"), aspect);
            rd.SaveGDAL(os.path.join(os.path.dirname(drm_filepath),"SlopeInitialRes.tif"), slope);
        else:
            print("AOI shp file" +shpfilepath + "is not inside of DEM"+drm_filepath+". Stopping.");
            return -1;
            #input('Press Enter for exit...')
            #exit;    
        
        
        
        #reopening SRTM products
        #read srtm products
        aspect_gdal_object = gdal.Open(os.path.join(os.path.dirname(drm_filepath),"aspectInitialRes.tif"))      #aspect
        aspect_band = aspect_gdal_object.GetRasterBand(1)
        aspect_band_array = aspect_band.ReadAsArray() 
        
        slope_gdal_object = gdal.Open(os.path.join(os.path.dirname(drm_filepath),"SlopeInitialRes.tif"))        #slope
        slope_band = slope_gdal_object.GetRasterBand(1)
        slope_band_array = slope_band.ReadAsArray() 
        
        #get PRODUCTS spatial resolution
        srtm_gt,srtm_xsize,srtm_ysize,srtm_ext,srtm_dpx,srtm_dpy=getGeotiffParams(aspect_gdal_object);
            
        
        #check if SRTM products inside of SHP AOI ad crop it
        if check_shp_inside_raster(srtm_ext,shp_extent):
            #do image crop
            aspect_cropped,ColMinInd,RowMinInd =crop_by_shp(shp_extent,srtm_ext,srtm_dpx,srtm_dpy,aspect_band_array)
            slope_cropped,ColMinInd,RowMinInd =crop_by_shp(shp_extent,srtm_ext,srtm_dpx,srtm_dpy,slope_band_array)
            
            #for testing purporses 
            saveGeoTiff(slope_cropped,'test_crop_slope.tif',slope_gdal_object,ColMinInd,RowMinInd) #tryna save cropped geotiff
        
        else:
            print("SRTM is outside of the AOI, exiting...")
            return -1;
            #exit();
    
    was_corrected=False; #flag to check if resolution and scale were corrected to landsat8        
    #STEP 1. CROP geotiffs one by one with AOI shape file
    print("Step. 1. Starting geotiff crop operation...")        
    for myfile in file_for_crop:
        #read geotiff
        gdal_object = gdal.Open(os.path.join(in_path,myfile))
        band = gdal_object.GetRasterBand(1)
        band_array = band.ReadAsArray() 
        
        #get spatial resolution
        #do image crop
        gt,xsize,ysize,ext,dpx,dpy=getGeotiffParams(gdal_object)
        """
        gt=gdal_object.GetGeoTransform()
        xsize = gdal_object.RasterXSize
        ysize = gdal_object.RasterYSize #x and y raster size in pixels
        ext=GetExtent(gt,ysize,xsize) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
        #resolution in meters
        dpx=(ext[3][0]-ext[0][0])/xsize
        dpy=(ext[0][1]-ext[2][1])/ysize
        print(ext)
        """
        #apply shp file
        #try:
        #    shp_extent=get_shp_extent(shpfilepath);
        #except:
        #    print("Can not read shp AOI file.")
        
        #check shp posiiton inside of tiff
        if check_shp_inside_raster(ext,shp_extent):
            #do image crop
            sampleImage,ColMinInd,RowMinInd =crop_by_shp(shp_extent,ext,dpx,dpy,band_array)
            
        else:
            print("AOI shp file" +shpfilepath + "is not inside of tiff"+myfile+". Stopping.");
            #input('Press Enter for exit...')
            return -1;
            #exit;
            
        #topocorrection
        if is_topocorrection==True: #topocorrection flag    
            if was_corrected==False:
                print('compute slope and aspect cropped')
                #коррекция aspect по Landsat8
                #adjust srtm resolution to landsat8
                [hlc,wlc]=np.shape(sampleImage);
                aspect_band_cropped=resize(aspect_cropped,(hlc,wlc),preserve_range=True,mode="wrap") #it works with scikit-image resize
        
                #коррекция slope по Landsat8
                slope_band_cropped=resize(slope_cropped,(hlc,wlc),preserve_range=True,mode="wrap") #it works with scikit-image resize
                
                
                Cos_i=np.cos(np.deg2rad(slope_band_cropped))*np.cos(np.deg2rad(SolarZenith))+\
                np.sin(np.deg2rad(slope_band_cropped))*np.sin(np.deg2rad(SolarZenith))*\
                np.cos(np.deg2rad(SunAzimuth-aspect_band_cropped));
                
                #ЭТОТ РАСЧЕТ КОС I РАССМАТРИВАЕТ ВСЕ СКЛОНЫ КАК ОСВЕЩЕННЫЕ ПОД ПРЯМЫМ УГЛОМ!                                                                            
                #Cos_i=np.cos(np.deg2rad(SolarZenith-slope_band_cropped));
                            
                #Do SCS+C correction anyway
                """
                print("Check correlation between Cos(i) and Luminocity")
                R_mat=np.corrcoef(Cos_i.ravel(),sampleImage.ravel()) 
                print("R="+str(R_mat[0,1]));            
                if( R_mat[0,1]<0.5):
                    print("No or weak correlation, use SCS algoritm...");
                    C=0;
                else:
                    print("Not a weak correlation, use SCS+C algoritm...");
                    (b,a)=np.polyfit(Cos_i.ravel(),sampleImage.ravel(),1);
                    C=a/b;
                 """
                (b,a)=np.polyfit(Cos_i.ravel(),sampleImage.ravel(),1);
                C=a/b;                                                                   
                was_corrected=True; #switch the flag to true                                                                                        
            
            print("Performing topographic correction.. Please, WAIT..")
            #Sun-Canopy-Sensor Correction (SCS)+C
            band_array=np.uint16(sampleImage*\
                    ((np.cos(np.deg2rad(SolarZenith))*np.cos(np.deg2rad(slope_band_cropped))+C)\
                     /(C+Cos_i)));
            #pic_show(sampleImage,"landsat initial");
            #hist_show(sampleImage);
            #pic_show(band_array,"landsat SCS corrected");
            #hist_show(band_array);                       
        else: #no topocorrection
            print("No topocorrection was selected..")
            band_array=copy.copy(sampleImage); #no operation
                                                                                                    
        #check shp posiiton inside of tiff
        #if check_shp_inside_raster(ext,shp_extent):
        #    #do image crop
            #sampleImage,ColMinInd,RowMinInd =crop_by_shp(shp_extent,ext,dpx,dpy,band_array)
            
            #if is_topocorrection==True: #topocorrection flag  
            #    #do topocorrection with SCS Algorythm
                
                
                
        #drop image to the disk
        print("drop image to the disk")
        outfilename=os.path.join(dir_crop_path,"crop_"+myfile.lower());
        if not os.path.isdir(dir_crop_path):
            os.makedirs(dir_crop_path) #create output directory if none
        try:
            saveGeoTiff(band_array,outfilename,gdal_object,ColMinInd,RowMinInd) #save topocorrected Landsat crop
        except:
            print("Can not write on a disk... and/or error(s) in saveGeoTiff function")
              
        
    
            
    #STEP 2. COMPUTE pseudocolor RGB stacks and satellite indexes
    """
    автоопределение BANDs для дефолтных имен, если пользователь не задал имена (пока что имена по умолчанию), 
    пропускаем индекс или RGB стек, если не находим BAND NUMBER
    """
    print("Step. 2. Getting names of the cropped files...")        
    #getting names of the cropped files, aquire band names
    file_for_processing=[];
    try:
        for file in os.listdir(dir_crop_path): #набираем файлы из папки с кадрированными изображениями
            file=file.lower();
            if file.endswith("."+fileext.lower()):
                file_for_processing.append(file);
                print(file+" was added to the processing queue.");
    except(FileNotFoundError):
            print("Input image folder doesn\'t exist...");
    
    bands={};  #dictionary storing band names 
    for myfile in file_for_processing:
        for N in range(1,9):            
            #populating bands dictionary
            if band_number_inname.replace('%n%',str(N),1) in myfile:
                try:
                    gdal_object = gdal.Open(os.path.join(dir_crop_path,myfile)) #as new gdal_object was created, no more ColMinInd,RowMinInd
                    bands['band'+str(N)]=gdal_object.GetRasterBand(1).ReadAsArray() ;
                except:
                    print("Error! Can not read cropped bands!")
    #print("Bands dictionary output:")
    #print(bands) 
    
    try:
        #create RGB stacks:
        #truecolor
        if ('rgb' in bandStacks):
            truecolorRGB=image_stack(bands['band4'],bands['band3'],bands['band2'],do_norm8=1,do_show=0)  
        
        #Комбинация 7-4-2. Изображение близкое к естественным цветам, позволяет анализировать состояние атмосферы и дым. Здоровая растительность выглядит ярко зеленой, ярко розовые участки детектируют открытую почву, коричневые и оранжевые тона характерны для разреженной растительности.
        if ('742' in bandStacks):
            b742RGB=image_stack(bands['band7'],bands['band4'],bands['band2'],do_norm8=1,do_show=0)
        #Комбинация 5-4-1. Изображение близкое к предыдущему, позволяет анализировать сельскохозяйственные культуры
        if ('652' in bandStacks):
            b652RGB=image_stack(bands['band6'],bands['band5'],bands['band2'],do_norm8=1,do_show=0)
        #Комбинация 4-5-3. Изображение позволяет четко различить границу между водой и сушей, с большой точностью будут детектироваться водные объекты внутри суши. Эта комбинация отображает растительность в различных оттенках и тонах коричневого, зеленого и оранжевого, дает возможность анализа влажности и полезны при изучении почв и растительного покрова.
        if ('453' in bandStacks):
            b453RGB=image_stack(bands['band4'],bands['band5'],bands['band3'],do_norm8=1,do_show=0)
        
        #after Aydal, 2007
        if ('642' in bandStacks):
            b642RGB=image_stack(bands['band6'],bands['band4'],bands['band2'],do_norm8=1,do_show=0)   
        if ('765' in bandStacks):
            b765RGB=image_stack(bands['band7'],bands['band6'],bands['band5'],do_norm8=1,do_show=0)   
        if ('764' in bandStacks):
            b764RGB=image_stack(bands['band7'],bands['band6'],bands['band4'],do_norm8=1,do_show=0)   
        
        
        #create indexes
        if ('NDVI' in products):
            NDVI=(bands['band5']-bands['band4'])/(bands['band5']+bands['band4']) #NDVI
        if ('IOA' in products) or ('CA' in products):
            IOA=(bands['band4']/bands['band2']) #Iron oxide alteration [Doğan Aydal, 2007]
        if ('HA' in products) or ('CA' in products):
            HA=(bands['band7']/bands['band2'])#Hydroxyl alteration [Doğan Aydal, 2007]
        if ('CM' in products):
            CM=(bands['band7']/bands['band6']) #Clay minerals [Doğan Aydal, 2007]
        
        #compute PCA 
        if ('PC' in products):
            print("Started to compute PCA...")
            print("Flatten image matrix...")
            flattened_img_matrix=mat4pca((bands['band1'],bands['band2'],bands['band3'],\
                                          bands['band4'],bands['band5'],bands['band6'],bands['band7']))
            #mybands=[bands['band1'],bands['band2'],bands['band3'],\
            #                              bands['band4'],bands['band5'],bands['band6'],bands['band7']]
            #tmp_matrix=[mynormalize16to8(tmpband) for tmpband in mybands]
            #flattened_img_matrix=mat4pca(tmp_matrix) #same but images are normalized to uint8
            print("Compute PCA matrix, the variance and the mean...")
            
            m,n=np.shape(bands['band3']) #temporary height and width
            (pca,eigenvalues,var_X,mean_X)=pca_make(flattened_img_matrix,7,m,n)
        
        
        #create cumulative image composite image of the hydroxyl image(red band), the iron oxide image
        #(green band) and the average of these two images (blue band).
        if ('CA' in products):
            index_composite=image_stack(HA,IOA,(HA+IOA)/2,1,0)
    
    except:
        print('No bands or bands error!');
        return -1;
    #GENERAL OUTPUT
    if ('PC' in products):
        print("Prepare to show PCA images")
        
        #later incorporate path into functions
        if not os.path.isdir(dir_products_path):
                    os.makedirs(dir_products_path) #create output products directory if none
                    
        fig_save_cumsum_path=os.path.join(dir_products_path,"variance_cumsum.svg");
        fig_save_pca_path=os.path.join(dir_products_path,"pca_comp.png");
        
        #num_comp=show_pca_cumsum(pca,fig_save_cumsum_path); #pca variance cumsum to determine right number of components
        #show_pca_images(eigenvalues,mean_X,m,n,fig_save_pca_path) #show pca component images
    
    #COMPUTE Landsat and PCA stat for the CROSTA METHOD
    try:
        stat_bands_save=os.path.join(dir_products_path,"bands_stat.xls");
        cor_bands_save=os.path.join(dir_products_path,"bands_cor_stat.xls");
        cov_bands_pca_save=os.path.join(dir_products_path,"bands_pca_cov_stat.xls");
        
        print("Saving band stat to {}".format(stat_bands_save));
        save_landsat_bands_stat(bands,stat_bands_save);
        
        print("Saving bands mutual correlation to {}".format(cor_bands_save));
        save_landsat_mutual_cor(bands,cor_bands_save);
    except:
        print('can not save band stats')
    try:                #correlation of bands and PCA comp may be potentially errorneous, dep on PCA number
        print("Saving covariance between bands and PCA components to {}".format(cov_bands_pca_save));
        save_landsat_pca_cov(bands,eigenvalues,cov_bands_pca_save);
    except:
        print('Can not compute/save pca/bands covariance...')
        
    
    #save RGB's and index to the disk
    print("Saving products on a disk")
    if not os.path.isdir(dir_products_path):
        os.makedirs(dir_products_path) #create output directory if none
    try:
        print("Saving RGBs...")
        ColMinInd=0; RowMinInd=0; #because we work on already cropped pictures
        if ('rgb' in bandStacks):
            saveGeoTiff(truecolorRGB,os.path.join(dir_products_path,"truecolorRGB"+".tif"),gdal_object,ColMinInd,RowMinInd);     
        if ('742' in bandStacks):
            saveGeoTiff(b742RGB,os.path.join(dir_products_path,"b742RGB"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('652' in bandStacks):
            saveGeoTiff(b652RGB,os.path.join(dir_products_path,"b652RGB"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('453' in bandStacks):
            saveGeoTiff(b453RGB,os.path.join(dir_products_path,"b453RGB"+".tif"),gdal_object,ColMinInd,RowMinInd);
         #Aydal pseudocolor:
        if ('642' in bandStacks):
            saveGeoTiff(b642RGB,os.path.join(dir_products_path,"b642RGB"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('765' in bandStacks):
            saveGeoTiff(b765RGB,os.path.join(dir_products_path,"b765RGB"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('764' in bandStacks):
            saveGeoTiff(b764RGB,os.path.join(dir_products_path,"b764RGB"+".tif"),gdal_object,ColMinInd,RowMinInd);
        
        print("Saving Indexes...")
        if ('NDVI' in products):
            saveGeoTiff(NDVI,os.path.join(dir_products_path,"NDVI"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('IOA' in products):
             saveGeoTiff(IOA,os.path.join(dir_products_path,"IOA"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('HA' in products):
            saveGeoTiff(HA,os.path.join(dir_products_path,"HA"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('CM' in products):
            saveGeoTiff(CM,os.path.join(dir_products_path,"CM"+".tif"),gdal_object,ColMinInd,RowMinInd);
        if ('CA' in products):
            saveGeoTiff(index_composite,os.path.join(dir_products_path,"CumulativeAlteration"+".tif"),gdal_object,ColMinInd,RowMinInd);
        
        
        
        if ('PC' in products):
            print("Saving PCA components...")    
            print("Result for the RANDOMIZED solver")
            for ev in range(0,len(eigenvalues[:,1,1])):
                PCAcomp=eigenvalues[ev,:,:].reshape(m,n);
                saveGeoTiff(PCAcomp,os.path.join(dir_products_path,"PCA{}_".format(ev+1)+".tif"),gdal_object,ColMinInd,RowMinInd);    
            
        print("Products data were saved.")
        return 1
    except:
        print("Can not write PRODUCTS on a disk... and/or error(s) in saveGeoTiff function")
        return -1
    
    print("Operations were finished. It took {} sec".format(time.time()-time_start))
